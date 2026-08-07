# %% [markdown]
# # exp342 exp226 Student-t residual-offset emission audit
#
# Stage 0 replaces only the Gaussian residual family with fixed df=4 Student-t.
# It freezes Student-t scores beside the SHA-pinned saved exp280 Gaussian
# control, then attaches truth for rank, stress, circular-control, and
# extreme-residual readouts. Stage 1 keeps the exp281 exact-HMM contract fixed
# and replaces only the Gaussian emission with df=4 Student-t. Inference and
# submission stay disabled.

# %% [markdown]
# ## Contents
# 1. Imports and fixed experiment contract
# 2. Runtime, configuration, path, and SHA helpers
# 3. Saved exp280 control and exp226/raw input checks
# 4. Fixed df=4 Student-t target-free scoring
# 5. Target-free score bundle freeze and circular controls
# 6. Truth-only block labels and persistent-offset readout
# 7. Fold, stress, extreme-residual metrics and fixed gate
# 8. Stage 0 Kaggle CPU orchestration and artifact guards
# 9. Stage 1 exact-HMM kernel and Student-t path generation
# 10. Stage 1 saved-parent evaluation and artifact guards
# 11. Setup and contract preview
# 12. Run the explicitly selected stage

# %%
from __future__ import annotations

import gc
import gzip
import hashlib
import json
import math
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

try:
    import numba
    from numba import njit, prange, set_num_threads

    NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - Kaggle includes numba.
    numba = None
    NUMBA_AVAILABLE = False

    def njit(*args: Any, **kwargs: Any):  # type: ignore[misc]
        def decorator(function):
            return function

        return decorator

    def prange(*args: Any):  # type: ignore[misc]
        return range(*args)

    def set_num_threads(_: int) -> None:
        return None


EXPERIMENT_NAME = "exp342_exp226_student_t_residual_offset_emission_audit"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
EXPECTED_SHIFTS = np.asarray(
    [-80.0, -40.0, -20.0, -10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0],
    dtype=np.float64,
)


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP342_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


# %% [markdown]
# ## 2. Runtime, configuration, path, and SHA helpers


# %%
def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        item = float(value)
        return item if math.isfinite(item) else None
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def get_nested(config: dict[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


def load_experiment_config() -> dict[str, Any]:
    root = project_root()
    candidates = (
        Path.cwd() / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    )
    for path in candidates:
        config = read_yaml(path)
        if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
            return config
    raise FileNotFoundError(f"exp342 config not found in {[str(path) for path in candidates]}")


def artifact_dir() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        output = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        output = project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    return output


def metrics_output_path() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        return KAGGLE_WORKING_ROOT / "metrics.json"
    return project_root() / "experiments" / EXPERIMENT_NAME / "metrics.json"


def train_data_dir(config: dict[str, Any]) -> Path:
    if KAGGLE_INPUT_ROOT.exists():
        fixed = (
            KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction" / "train",
            KAGGLE_INPUT_ROOT / "competitions" / "rogii-wellbore-geology-prediction" / "train",
        )
        for candidate in fixed:
            if next(candidate.glob("*__horizontal_well.csv"), None) is not None:
                return candidate
        for candidate in sorted(KAGGLE_INPUT_ROOT.glob("**/train")):
            if next(candidate.glob("*__horizontal_well.csv"), None) is not None:
                return candidate
    return project_root() / str(get_nested(config, "data.train_dir") or "data/raw/train")


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_decompressed(path: str | Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapping_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def dataframe_content_sha(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    chosen = list(frame.columns) if columns is None else list(columns)
    digest = hashlib.sha256()
    for column in chosen:
        digest.update(column.encode())
        values = frame[column]
        if pd.api.types.is_numeric_dtype(values):
            array = np.ascontiguousarray(values.to_numpy())
            digest.update(str(array.dtype).encode())
            digest.update(array.tobytes())
        else:
            for value in values.astype(str):
                digest.update(value.encode())
                digest.update(b"\n")
    return digest.hexdigest()


def write_csv_gzip(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = frame.to_csv(index=False).encode()
    path.write_bytes(gzip.compress(payload, compresslevel=6, mtime=0))
    return {
        "path": str(path),
        "rows": len(frame),
        "raw_sha256": sha256_path(path),
        "decompressed_sha256": hashlib.sha256(payload).hexdigest(),
        "content_sha256": dataframe_content_sha(frame),
    }


def resolve_existing(filename: str, candidates: Iterable[str]) -> Path:
    root = project_root()
    checked: list[str] = []
    for raw in candidates:
        candidate = Path(str(raw))
        for path in (candidate, root / candidate, Path.cwd() / candidate):
            checked.append(str(path))
            if path.exists() and path.is_file():
                return path
    if KAGGLE_INPUT_ROOT.exists():
        for path in sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}")):
            if path.is_file():
                return path
    raise FileNotFoundError(f"could not resolve {filename}; checked={checked}")


def resolve_pattern_file(filename: str, patterns: Iterable[str]) -> Path:
    root = project_root()
    checked: list[str] = []
    for raw in patterns:
        pattern = str(raw)
        if "*" not in pattern:
            candidate = Path(pattern)
            for path in (candidate, root / candidate, Path.cwd() / candidate):
                checked.append(str(path))
                if path.exists() and path.is_file():
                    return path
            continue
        for base in (root, Path.cwd(), KAGGLE_INPUT_ROOT):
            if not base.exists():
                continue
            glob_pattern = pattern
            if pattern.startswith("**/"):
                glob_pattern = pattern
            elif Path(pattern).is_absolute():
                continue
            for path in sorted(base.glob(glob_pattern)):
                checked.append(str(path))
                if path.is_file():
                    return path
    if KAGGLE_INPUT_ROOT.exists():
        for path in sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}")):
            if path.is_file():
                return path
    raise FileNotFoundError(f"could not resolve {filename}; checked={checked}")


def stable_uint64(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def stable_nonzero_rotation(well_id: str, block_id: int, candidate_count: int) -> int:
    if candidate_count < 2:
        return 0
    return 1 + stable_uint64(EXPERIMENT_NAME, "circular_shift_bank", well_id, block_id) % (
        candidate_count - 1
    )


def rank_descending(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("ranking requires one finite score per shift")
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int16)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.int16)
    return ranks


def student_t_log_likelihood(zscore: np.ndarray, degrees_of_freedom: float) -> np.ndarray:
    values = np.asarray(zscore, dtype=np.float64)
    df = float(degrees_of_freedom)
    if not math.isfinite(df) or df <= 0.0:
        raise ValueError("Student-t degrees_of_freedom must be positive and finite")
    output = -0.5 * (df + 1.0) * np.log1p(np.square(values) / df)
    if not np.isfinite(output).all():
        raise ValueError("Student-t log likelihood must be finite")
    return output


def validate_scientific_contract(
    config: dict[str, Any], *, require_run_approval: bool = False
) -> None:
    shifts = [float(value) for value in get_nested(config, "audit.shift_bank_ft") or []]
    emission = get_nested(config, "audit.student_t_emission") or {}
    guards = get_nested(config, "validation.stage_0_pass_requires_all") or {}
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp342 route must remain pf_beam")
    if get_nested(config, "lineage.parent") != (
        "exp281_exp226_residual_offset_exact_hmm_transition_probe"
    ):
        raise ValueError("exp342 Stage 1 parent must remain exp281")
    if get_nested(config, "lineage.stage_0_source") != (
        "exp280_exp226_shift_likelihood_separability_readout"
    ):
        raise ValueError("exp342 Stage 0 source must remain exp280")
    if shifts != EXPECTED_SHIFTS.tolist():
        raise ValueError("exp342 fixes the approved 13-value shift bank")
    if int(get_nested(config, "audit.block_rows") or 0) != 512:
        raise ValueError("exp342 fixes non-overlapping 512-row blocks")
    if (
        get_nested(config, "audit.block_policy")
        != "non_overlapping_from_suffix_start_keep_short_tail"
    ):
        raise ValueError("exp342 fixes the non-overlapping short-tail block policy")
    if get_nested(config, "audit.score_aggregation") != "mean_row_log_likelihood":
        raise ValueError("exp342 fixes mean row log-likelihood aggregation")
    if get_nested(config, "audit.tie_policy") != "config_shift_bank_order":
        raise ValueError("exp342 fixes config-order tie resolution")
    if emission.get("kind") != "student_t_df4_raw_gr":
        raise ValueError("exp342 fixes the Student-t raw-GR emission")
    if float(emission.get("degrees_of_freedom", 0.0)) != 4.0:
        raise ValueError("exp342 fixes Student-t df=4")
    if [float(value) for value in emission.get("sigma_clip", [])] != [10.0, 60.0]:
        raise ValueError("exp342 fixes GR sigma clip [10, 60]")
    if emission.get("additional_clip") != "none":
        raise ValueError("exp342 forbids Student-t likelihood clipping")
    if float(get_nested(config, "model.student_t.degrees_of_freedom") or 0.0) != 4.0:
        raise ValueError("exp342 model contract fixes Student-t df=4")
    if float(guards.get("minimum_pooled_mrr_gain", -1.0)) != 0.01:
        raise ValueError("exp342 fixes pooled MRR gain at 0.01")
    if float(guards.get("minimum_pooled_top3_gain", -1.0)) != 0.01:
        raise ValueError("exp342 fixes pooled top3 gain at 0.01")
    if int(guards.get("minimum_improved_folds_mrr", 0)) != 4:
        raise ValueError("exp342 fixes MRR fold gate at 4/5")
    if int(guards.get("minimum_improved_folds_top3", 0)) != 4:
        raise ValueError("exp342 fixes top3 fold gate at 4/5")
    extreme = guards.get("extreme_residual") or {}
    if float(extreme.get("absolute_z_threshold", 0.0)) != 3.0:
        raise ValueError("exp342 fixes extreme residual threshold |z|>=3")
    if int(extreme.get("minimum_extreme_rows_per_block", 0)) != 1:
        raise ValueError("exp342 fixes extreme block membership at one or more rows")
    required_true = (
        "require_stress_mrr_non_regression",
        "require_stress_top3_non_regression",
        "require_real_shuffle_mrr_gap_not_worse",
        "require_real_shuffle_top3_gap_not_worse",
    )
    if not all(bool(guards.get(key)) for key in required_true):
        raise ValueError("exp342 requires every preregistered stress/control gate")
    if not bool(extreme.get("require_top3_improvement")) or not bool(
        extreme.get("require_mean_regret_improvement")
    ):
        raise ValueError("exp342 requires both extreme-residual gates")
    if get_nested(config, "model.stage_0.gaussian_control") != "sha_pinned_saved_exp280":
        raise ValueError("exp342 must reuse the saved exp280 Gaussian control")
    if int(get_nested(config, "model.stage_0.active_variant_count") or 0) != 1:
        raise ValueError("exp342 Stage 0 fixes one scientific variant")
    if get_nested(config, "model.stage_0.active_variants") != ["student_t_df4"]:
        raise ValueError("exp342 Stage 0 variant must remain student_t_df4")
    if int(get_nested(config, "execution_contract.stage_0.scientific_scores") or 0) != 1:
        raise ValueError("exp342 Stage 0 fixes one scientific score")
    if int(get_nested(config, "execution_contract.stage_0.control_scores") or 0) != 1:
        raise ValueError("exp342 Stage 0 fixes one saved control score")
    if bool(get_nested(config, "execution_contract.parent_control_retraining")):
        raise ValueError("exp342 forbids parent/control regeneration")
    if int(get_nested(config, "execution_contract.stage_1_if_pass.hmm_well_runs") or 0) != 773:
        raise ValueError("exp342 Stage 1 reservation must remain 773 HMM well-runs")
    expected_zero = {
        "execution_contract.stage_0.model_configs": 0,
        "execution_contract.stage_0.trained_folds": 0,
        "execution_contract.stage_0.boosters": 0,
        "execution_contract.stage_0.hmm_well_runs": 0,
    }
    for key, expected in expected_zero.items():
        if int(get_nested(config, key) or 0) != expected:
            raise ValueError(f"exp342 requires {key}={expected}")
    stage_1_implemented = bool(get_nested(config, "implementation.stage_1_implemented"))
    forbidden_true = (
        "execution.run_inference",
        "execution.create_submission",
        "inference.enabled",
        "inference.create_submission",
        "implementation.inference_enabled",
        "implementation.submission_enabled",
    )
    if any(bool(get_nested(config, key)) for key in forbidden_true):
        raise ValueError("exp342 inference and submission must remain disabled")
    if not bool(get_nested(config, "implementation.enabled")) or not bool(
        get_nested(config, "execution.implementation_approved")
    ):
        raise ValueError("exp342 Stage 0 implementation approval is required")
    if stage_1_implemented:
        if not bool(get_nested(config, "execution.stage_1_override_approved")):
            raise ValueError("exp342 Stage 1 requires the explicit post-Stage-0 override")
        if bool(get_nested(config, "execution.run_stage_0")):
            raise ValueError("exp342 Stage 0 must not rerun during Stage 1")
        validate_stage_1_contract(config)
    elif bool(get_nested(config, "execution.run_stage_1")):
        raise ValueError("exp342 Stage 1 cannot run before implementation")
    if require_run_approval:
        approved = bool(get_nested(config, "execution.kaggle_push_approved")) and bool(
            get_nested(config, "runtime.kaggle.train_run_on_push")
        )
        if stage_1_implemented:
            approved = approved and bool(get_nested(config, "execution.run_stage_1"))
            message = "exp342 Stage 1 package/push/run is not approved"
        else:
            approved = approved and bool(get_nested(config, "execution.run_stage_0"))
            message = "exp342 Stage 0 package/push/run is not approved"
        if not approved:
            raise RuntimeError(message)


def validate_stage_1_contract(config: dict[str, Any]) -> None:
    hmm = get_nested(config, "model.hmm") or {}
    fixed = {
        "delta_min_ft": -80.0,
        "delta_max_ft": 80.0,
        "step": 0.35,
        "n_rates": 41,
        "rate_span": 0.10,
        "sig_r": 0.002,
        "sig_p": 0.02,
        "emission": "student_t_df4",
        "student_t_degrees_of_freedom": 4.0,
        "lam": 1.0,
        "sigma_mode": "std",
        "start_delta_ft": 0.0,
        "start_sig": 0.75,
        "initial_offset_rate": 0.0,
        "r0_sig": 0.01,
        "mom": 0.998,
        "rate_center": "zero",
        "additional_likelihood_clip": "none",
        "typewell_extension_ft": 40.0,
        "transition_center": "exp226_tvt_geop_row_delta",
    }
    for key, expected in fixed.items():
        if hmm.get(key) != expected:
            raise ValueError(
                f"exp342 Stage 1 fixes model.hmm.{key}={expected!r}, got {hmm.get(key)!r}"
            )
    if hmm.get("active_variants") != [
        "student_t_df4_residual_offset_delta80_step035_rate41"
    ]:
        raise ValueError("exp342 Stage 1 fixes one Student-t exact-HMM variant")
    contract = get_nested(config, "execution_contract.stage_1_override") or {}
    expected_counts = {
        "scientific_variants": 1,
        "hmm_well_runs": 773,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
    }
    if contract != expected_counts:
        raise ValueError(f"exp342 Stage 1 execution contract changed: {contract}")
    guards = get_nested(config, "validation.stage_1_pass_requires_all") or {}
    expected_guards = {
        "minimum_rmse_gain_vs_exp281_ft": 0.05,
        "minimum_improved_folds": 4,
        "maximum_scope_rmse_regression_ft": 0.0,
        "maximum_by_well_p95_delta_rmse_ft": 0.0,
        "maximum_worst_well_regression_ft": 0.25,
        "required_parent_rmse_parity_atol_ft": 0.00001,
        "required_finite_coverage": 1.0,
        "required_row_identity_coverage": 1.0,
        "maximum_direct_rmse_for_promotion": 9.427109596582213,
    }
    for key, expected in expected_guards.items():
        if float(guards.get(key, float("nan"))) != expected:
            raise ValueError(f"exp342 Stage 1 gate changed at {key}")
    if guards.get("required_scopes") != [
        "long_tail_1000_plus",
        "hidden_like_spatial",
        "hidden_like_typewell_purged",
    ]:
        raise ValueError("exp342 Stage 1 fixes the three preregistered stress scopes")


# %% [markdown]
# ## 3. Saved exp280 control and exp226/raw input checks


# %%
def load_exp280_gaussian_control(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    spec = get_nested(config, "data.exp280_source") or {}
    score_path = resolve_pattern_file(
        str(spec["score_filename"]), [str(value) for value in spec["score_patterns"]]
    )
    decompressed_sha = sha256_gzip_decompressed(score_path)
    if decompressed_sha != str(spec["score_decompressed_sha256"]):
        raise ValueError("exp280 Gaussian score decompressed SHA changed")
    contract_path = resolve_pattern_file(
        str(spec["contract_filename"]),
        [str(value) for value in spec["contract_patterns"]],
    )
    contract = json.loads(contract_path.read_text())
    if bool(contract.get("truth_attached")):
        raise ValueError("exp280 Gaussian score contract must be truth-free")
    if contract.get("target_free_score_content_sha256") != str(spec["score_content_sha256"]):
        raise ValueError("exp280 Gaussian score content declaration changed")
    if contract.get("scientific_contract_sha256") != str(spec["scientific_contract_sha256"]):
        raise ValueError("exp280 scientific contract SHA changed")
    if list(map(float, contract.get("shift_bank_ft", []))) != EXPECTED_SHIFTS.tolist():
        raise ValueError("exp280 Gaussian shift bank changed")
    if int(contract.get("block_rows", -1)) != 512:
        raise ValueError("exp280 Gaussian block contract changed")

    scores = pd.read_csv(score_path, dtype={"well_id": str})
    forbidden = {"tvt_true", "tvt_pred", "gr_delta", "error", "abs_error", "TVT"}
    leaked = sorted(forbidden.intersection(scores.columns))
    if leaked:
        raise ValueError(f"exp280 control unexpectedly contains truth/error columns: {leaked}")
    required = {
        "well_id",
        "fold",
        "block_id",
        "block_start_suffix_offset",
        "block_end_suffix_offset",
        "block_start_row_idx",
        "block_end_row_idx",
        "block_row_count",
        "md_since_min_ft",
        "md_since_max_ft",
        "md_since_mid_ft",
        "observed_gr_share",
        "shift_slot",
        "shift_ft",
        "likelihood_mean",
        "likelihood_sum",
        "likelihood_rank",
    }
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"exp280 Gaussian score table missing {missing}")
    integer_columns = (
        "fold",
        "block_id",
        "block_start_suffix_offset",
        "block_end_suffix_offset",
        "block_start_row_idx",
        "block_end_row_idx",
        "block_row_count",
        "shift_slot",
        "likelihood_rank",
    )
    for column in integer_columns:
        scores[column] = pd.to_numeric(scores[column], errors="raise").astype(np.int64)
    for column in required.difference({"well_id", *integer_columns}):
        scores[column] = pd.to_numeric(scores[column], errors="raise").astype(np.float64)
    scores["well_id"] = scores["well_id"].astype(str)
    scores = scores.sort_values(
        ["well_id", "block_id", "shift_slot"], kind="mergesort"
    ).reset_index(drop=True)
    expected_blocks = int(get_nested(config, "validation.expected_blocks"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(scores) != expected_blocks * len(EXPECTED_SHIFTS):
        raise ValueError("exp280 Gaussian score row count changed")
    if scores["well_id"].nunique() != expected_wells:
        raise ValueError("exp280 Gaussian score well count changed")
    group_size = scores.groupby(["well_id", "block_id"], sort=False).size()
    if not group_size.eq(len(EXPECTED_SHIFTS)).all() or len(group_size) != expected_blocks:
        raise ValueError("each exp280 block must contain exactly 13 shifts")
    observed_shifts = (
        scores["shift_ft"].to_numpy(np.float64).reshape(expected_blocks, len(EXPECTED_SHIFTS))
    )
    if not np.array_equal(observed_shifts, np.broadcast_to(EXPECTED_SHIFTS, observed_shifts.shape)):
        raise ValueError("exp280 Gaussian shift identity/order changed")
    gaussian = (
        scores["likelihood_mean"]
        .to_numpy(np.float64)
        .reshape(expected_blocks, len(EXPECTED_SHIFTS))
    )
    stored_ranks = (
        scores["likelihood_rank"].to_numpy(np.int64).reshape(expected_blocks, len(EXPECTED_SHIFTS))
    )
    recomputed_ranks = np.vstack([rank_descending(row) for row in gaussian])
    if not np.array_equal(stored_ranks, recomputed_ranks):
        raise ValueError("exp280 Gaussian stored ranks no longer match saved scores")
    evidence = [
        {
            "name": "exp280_saved_gaussian_scores",
            "path": str(score_path),
            "raw_sha256": sha256_path(score_path),
            "decompressed_sha256": decompressed_sha,
            "declared_content_sha256": contract["target_free_score_content_sha256"],
            "rows": len(scores),
        },
        {
            "name": "exp280_saved_gaussian_contract",
            "path": str(contract_path),
            "raw_sha256": sha256_path(contract_path),
            "scientific_contract_sha256": contract["scientific_contract_sha256"],
        },
    ]
    return scores, evidence


def load_exp226_safe(config: dict[str, Any]) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    spec = get_nested(config, "data.exp226_oof") or {}
    path = resolve_existing(str(spec["filename"]), [str(value) for value in spec["candidates"]])
    actual_decompressed_sha = sha256_gzip_decompressed(path)
    expected_decompressed_sha = str(spec["expected_decompressed_sha256"])
    if actual_decompressed_sha != expected_decompressed_sha:
        raise ValueError(
            "exp226 decompressed SHA mismatch: "
            f"{actual_decompressed_sha} != {expected_decompressed_sha}"
        )
    safe_columns = [str(value) for value in spec["safe_columns"]]
    frame = pd.read_csv(path, usecols=safe_columns, dtype={"well_id": str})
    frame["well_id"] = frame["well_id"].astype(str)
    for column in ("row_idx", "suffix_offset", "fold"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.int64)
    frame["tvt_geop"] = pd.to_numeric(frame["tvt_geop"], errors="raise").astype(np.float64)
    frame = frame.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)
    if frame.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("exp226 safe OOF has duplicate well_id/row_idx")
    if not np.isfinite(frame["tvt_geop"]).all():
        raise ValueError("exp226 tvt_geop must be finite")
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    if len(frame) != expected_rows or frame["well_id"].nunique() != expected_wells:
        raise ValueError("exp226 row/well coverage does not match the fixed contract")
    if sorted(frame["fold"].unique().tolist()) != expected_folds:
        raise ValueError("exp226 fold set does not match the fixed contract")
    fold_counts = frame.groupby("well_id")["fold"].nunique()
    if not bool((fold_counts == 1).all()):
        raise ValueError("each exp226 well must belong to exactly one fold")
    manifest = {
        "name": "exp226_group_safe_oof_safe_columns",
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": sha256_path(path),
        "decompressed_sha256": actual_decompressed_sha,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "folds": sorted(int(value) for value in frame["fold"].unique()),
        "safe_columns": safe_columns,
    }
    return frame, path, manifest


def load_exp226_truth(
    path: Path,
    config: dict[str, Any],
    *,
    frozen_score_content_sha256: str,
) -> pd.DataFrame:
    if not frozen_score_content_sha256:
        raise ValueError("truth attachment requires a frozen target-free score content SHA")
    spec = get_nested(config, "data.exp226_oof") or {}
    truth_columns = [str(value) for value in spec["truth_columns"]]
    frame = pd.read_csv(path, usecols=truth_columns, dtype={"well_id": str})
    frame["well_id"] = frame["well_id"].astype(str)
    frame["row_idx"] = pd.to_numeric(frame["row_idx"], errors="raise").astype(np.int64)
    frame["tvt_true"] = pd.to_numeric(frame["tvt_true"], errors="raise").astype(np.float64)
    frame = frame.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)
    if frame.duplicated(["well_id", "row_idx"]).any() or not np.isfinite(frame["tvt_true"]).all():
        raise ValueError("exp226 truth readout rows must be unique and finite")
    return frame


def load_hidden_like_assignments(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.hidden_like") or {}
    if not bool(spec.get("enabled")):
        return pd.DataFrame(), {"enabled": False}
    path = resolve_existing(str(spec["filename"]), [str(value) for value in spec["candidates"]])
    actual_sha = sha256_path(path)
    if actual_sha != str(spec["expected_sha256"]):
        raise ValueError("hidden-like assignment SHA mismatch")
    frame = pd.read_csv(path, dtype={"well_id": str})
    required = {"well_id", *[str(value) for value in spec["role_columns"].values()]}
    if not required.issubset(frame.columns):
        raise ValueError(f"hidden-like assignments missing {sorted(required - set(frame.columns))}")
    if frame["well_id"].duplicated().any():
        raise ValueError("hidden-like assignments require one row per well")
    manifest = {
        "name": "exp115_hidden_like_fold_assignments",
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": actual_sha,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
    }
    return frame, manifest


def load_horizontal_without_truth(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=lambda column: column != "TVT")
    if "TVT" in frame.columns:
        raise ValueError("target-free horizontal reader must not expose TVT")
    return frame


# %% [markdown]
# ## 4. Fixed df=4 Student-t target-free scoring


# %%
def prepare_gr_inputs(
    horizontal_without_truth: pd.DataFrame,
    typewell: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    if "TVT" in horizontal_without_truth.columns:
        raise ValueError("target-free GR preparation forbids horizontal TVT")
    required_horizontal = {"MD", "GR", "TVT_input"}
    if not required_horizontal.issubset(horizontal_without_truth.columns):
        missing = sorted(required_horizontal - set(horizontal_without_truth.columns))
        raise ValueError(f"horizontal missing {missing}")
    if not {"TVT", "GR"}.issubset(typewell.columns):
        raise ValueError("typewell must contain TVT and GR")
    tw = typewell[["TVT", "GR"]].copy()
    tw["TVT"] = pd.to_numeric(tw["TVT"], errors="coerce")
    tw["GR"] = pd.to_numeric(tw["GR"], errors="coerce")
    tw = tw.dropna(subset=["TVT"]).sort_values("TVT", kind="mergesort")
    tw["GR"] = tw["GR"].ffill().bfill()
    if len(tw) < 2 or not np.isfinite(tw[["TVT", "GR"]].to_numpy()).all():
        raise ValueError("typewell requires at least two finite TVT/GR rows")
    typewell_tvt = tw["TVT"].to_numpy(np.float64)
    typewell_gr = tw["GR"].to_numpy(np.float64)
    known = horizontal_without_truth.loc[horizontal_without_truth["TVT_input"].notna()]
    if len(known) < 4:
        raise ValueError("well requires at least four known-prefix rows")
    known_tvt = pd.to_numeric(known["TVT_input"], errors="raise").to_numpy(np.float64)
    known_gr = pd.to_numeric(known["GR"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    typewell_at_known = np.interp(known_tvt, typewell_tvt, typewell_gr)
    residual = known_gr - typewell_at_known
    sigma_low, sigma_high = [
        float(value) for value in get_nested(config, "audit.student_t_emission.sigma_clip")
    ]
    gr_sigma = float(np.clip(np.nanstd(residual), sigma_low, sigma_high))
    if not np.isfinite(gr_sigma):
        raise ValueError("known-prefix GR residual sigma is not finite")
    gr_fill = float(np.nanmean(typewell_gr))
    all_gr = (
        pd.to_numeric(horizontal_without_truth["GR"], errors="coerce")
        .interpolate(limit_direction="both")
        .fillna(gr_fill)
        .to_numpy(np.float64)
    )
    return {
        "typewell_tvt": typewell_tvt,
        "typewell_gr": typewell_gr,
        "gr_sigma": gr_sigma,
        "all_gr_interpolated": all_gr,
        "known_rows": len(known),
        "known_residual_mean": float(np.mean(residual)),
        "known_residual_std_unclipped": float(np.std(residual)),
    }


def score_well_student_t_target_free(
    oof_safe: pd.DataFrame,
    horizontal_without_truth: pd.DataFrame,
    typewell: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    forbidden = set(
        str(value) for value in get_nested(config, "data.exp226_oof.forbidden_score_columns")
    )
    leaked = sorted(forbidden.intersection(oof_safe.columns))
    if leaked:
        raise ValueError(f"target-free score input contains forbidden exp226 columns: {leaked}")
    if "TVT" in horizontal_without_truth.columns:
        raise ValueError("target-free score input contains horizontal TVT")
    required_oof = {"well_id", "row_idx", "suffix_offset", "fold", "tvt_geop"}
    if not required_oof.issubset(oof_safe.columns):
        raise ValueError(f"safe OOF missing {sorted(required_oof - set(oof_safe.columns))}")
    oof = oof_safe.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    if oof.empty or oof["well_id"].nunique() != 1 or oof["fold"].nunique() != 1:
        raise ValueError("score_well_student_t_target_free requires one non-empty well and fold")
    row_idx = oof["row_idx"].to_numpy(np.int64)
    suffix_offset = oof["suffix_offset"].to_numpy(np.int64)
    if not np.array_equal(suffix_offset, np.arange(len(oof), dtype=np.int64)):
        raise ValueError("exp226 suffix_offset must be contiguous from zero")
    if row_idx.min() < 0 or row_idx.max() >= len(horizontal_without_truth):
        raise ValueError("exp226 row_idx is outside the raw horizontal frame")
    if horizontal_without_truth.iloc[row_idx]["TVT_input"].notna().any():
        raise ValueError("exp226 OOF rows must align only to unknown-suffix rows")

    prepared = prepare_gr_inputs(horizontal_without_truth, typewell, config)
    shifts = np.asarray(get_nested(config, "audit.shift_bank_ft"), dtype=np.float64)
    block_rows = int(get_nested(config, "audit.block_rows"))
    geop = oof["tvt_geop"].to_numpy(np.float64)
    candidate_tvt = geop[:, None] + shifts[None, :]
    expected_gr = np.empty_like(candidate_tvt)
    for slot in range(len(shifts)):
        expected_gr[:, slot] = np.interp(
            candidate_tvt[:, slot], prepared["typewell_tvt"], prepared["typewell_gr"]
        )
    raw_gr = prepared["all_gr_interpolated"][row_idx]
    zscore = (raw_gr[:, None] - expected_gr) / float(prepared["gr_sigma"])
    degrees_of_freedom = float(get_nested(config, "audit.student_t_emission.degrees_of_freedom"))
    log_likelihood = student_t_log_likelihood(zscore, degrees_of_freedom)
    extreme_threshold = float(
        get_nested(
            config,
            "validation.stage_0_pass_requires_all.extreme_residual.absolute_z_threshold",
        )
    )
    extreme = np.abs(zscore) >= extreme_threshold

    observed_gr = pd.to_numeric(horizontal_without_truth.iloc[row_idx]["GR"], errors="coerce")
    md = pd.to_numeric(horizontal_without_truth["MD"], errors="raise").to_numpy(np.float64)
    known_positions = np.flatnonzero(horizontal_without_truth["TVT_input"].notna().to_numpy())
    if not len(known_positions):
        raise ValueError("well has no known TVT_input prefix")
    last_known = int(known_positions[-1])
    md_since = md[row_idx] - md[last_known]
    block_id = suffix_offset // block_rows
    native = (candidate_tvt >= prepared["typewell_tvt"].min()) & (
        candidate_tvt <= prepared["typewell_tvt"].max()
    )
    extension = float(get_nested(config, "audit.typewell_extension_ft"))
    extended = (candidate_tvt >= prepared["typewell_tvt"].min() - extension) & (
        candidate_tvt <= prepared["typewell_tvt"].max() + extension
    )

    well = str(oof["well_id"].iloc[0])
    fold = int(oof["fold"].iloc[0])
    rows: list[dict[str, Any]] = []
    for block in np.unique(block_id):
        mask = block_id == block
        scores = log_likelihood[mask].mean(axis=0)
        score_sums = log_likelihood[mask].sum(axis=0)
        ranks = rank_descending(scores)
        extreme_counts = extreme[mask].sum(axis=0)
        max_abs_z = np.max(np.abs(zscore[mask]), axis=0)
        block_positions = np.flatnonzero(mask)
        for slot, shift in enumerate(shifts):
            rows.append(
                {
                    "well_id": well,
                    "fold": fold,
                    "block_id": int(block),
                    "block_start_suffix_offset": int(suffix_offset[block_positions[0]]),
                    "block_end_suffix_offset": int(suffix_offset[block_positions[-1]]),
                    "block_start_row_idx": int(row_idx[block_positions[0]]),
                    "block_end_row_idx": int(row_idx[block_positions[-1]]),
                    "block_row_count": int(mask.sum()),
                    "md_since_min_ft": float(np.min(md_since[mask])),
                    "md_since_max_ft": float(np.max(md_since[mask])),
                    "md_since_mid_ft": float(np.mean(md_since[mask])),
                    "observed_gr_share": float(observed_gr.iloc[block_positions].notna().mean()),
                    "shift_slot": int(slot),
                    "shift_ft": float(shift),
                    "student_t_likelihood_mean": float(scores[slot]),
                    "student_t_likelihood_sum": float(score_sums[slot]),
                    "student_t_likelihood_rank": int(ranks[slot]),
                    "extreme_abs_z_ge_3_count": int(extreme_counts[slot]),
                    "extreme_abs_z_ge_3_share": float(extreme_counts[slot] / mask.sum()),
                    "max_abs_z": float(max_abs_z[slot]),
                    "native_typewell_coverage": float(native[mask, slot].mean()),
                    "extended_typewell_coverage": float(extended[mask, slot].mean()),
                }
            )
    score_frame = pd.DataFrame(rows).sort_values(
        ["well_id", "block_id", "shift_slot"], kind="mergesort"
    )
    manifest = {
        "well_id": well,
        "fold": fold,
        "horizontal_rows": len(horizontal_without_truth),
        "evaluation_rows": len(oof),
        "blocks": int(block_id.max() + 1),
        "known_rows": int(prepared["known_rows"]),
        "last_known_row_idx": last_known,
        "gr_sigma": float(prepared["gr_sigma"]),
        "known_residual_mean": float(prepared["known_residual_mean"]),
        "known_residual_std_unclipped": float(prepared["known_residual_std_unclipped"]),
        "observed_eval_gr_share": float(observed_gr.notna().mean()),
        "student_t_score_finite_coverage": float(np.isfinite(log_likelihood).mean()),
        "degrees_of_freedom": degrees_of_freedom,
    }
    return score_frame.reset_index(drop=True), manifest


# %% [markdown]
# ## 5. Target-free score bundle freeze and circular controls


# %%
def build_target_free_score_bundle(
    student_scores: pd.DataFrame,
    gaussian_scores: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    keys = ["well_id", "block_id", "shift_slot"]
    student = student_scores.sort_values(keys, kind="mergesort").reset_index(drop=True)
    gaussian = gaussian_scores.sort_values(keys, kind="mergesort").reset_index(drop=True)
    if len(student) != len(gaussian):
        raise ValueError("Student-t and saved Gaussian score row counts differ")
    identity_columns = [
        "well_id",
        "fold",
        "block_id",
        "block_start_suffix_offset",
        "block_end_suffix_offset",
        "block_start_row_idx",
        "block_end_row_idx",
        "block_row_count",
        "shift_slot",
        "shift_ft",
    ]
    for column in identity_columns:
        left = student[column].to_numpy()
        right = gaussian[column].to_numpy()
        if pd.api.types.is_numeric_dtype(student[column]):
            equal = np.array_equal(left, right)
        else:
            equal = np.array_equal(left.astype(str), right.astype(str))
        if not equal:
            raise ValueError(f"Student-t/saved Gaussian identity mismatch: {column}")
    for column in (
        "md_since_min_ft",
        "md_since_max_ft",
        "md_since_mid_ft",
        "observed_gr_share",
    ):
        if not np.allclose(
            student[column].to_numpy(np.float64),
            gaussian[column].to_numpy(np.float64),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"Student-t/saved Gaussian metadata mismatch: {column}")

    bundle = student.copy()
    bundle["gaussian_likelihood_mean"] = gaussian["likelihood_mean"].to_numpy(np.float64)
    bundle["gaussian_likelihood_sum"] = gaussian["likelihood_sum"].to_numpy(np.float64)
    bundle["gaussian_likelihood_rank"] = gaussian["likelihood_rank"].to_numpy(np.int64)
    bundle["student_t_circular_likelihood_mean"] = np.nan
    bundle["student_t_circular_likelihood_rank"] = 0
    bundle["gaussian_circular_likelihood_mean"] = np.nan
    bundle["gaussian_circular_likelihood_rank"] = 0
    bundle["circular_rotation"] = 0

    for (well, block), positions in bundle.groupby(keys[:2], sort=True).indices.items():
        ordered_positions = np.asarray(positions, dtype=np.int64)
        ordered_positions = ordered_positions[
            np.argsort(
                bundle.iloc[ordered_positions]["shift_slot"].to_numpy(np.int64),
                kind="mergesort",
            )
        ]
        if len(ordered_positions) != len(EXPECTED_SHIFTS):
            raise ValueError(f"block {well}/{block} does not contain 13 shifts")
        rotation = stable_nonzero_rotation(str(well), int(block), len(ordered_positions))
        student_values = bundle.iloc[ordered_positions]["student_t_likelihood_mean"].to_numpy(
            np.float64
        )
        gaussian_values = bundle.iloc[ordered_positions]["gaussian_likelihood_mean"].to_numpy(
            np.float64
        )
        student_circular = np.roll(student_values, rotation)
        gaussian_circular = np.roll(gaussian_values, rotation)
        bundle.loc[ordered_positions, "student_t_circular_likelihood_mean"] = student_circular
        bundle.loc[ordered_positions, "student_t_circular_likelihood_rank"] = rank_descending(
            student_circular
        )
        bundle.loc[ordered_positions, "gaussian_circular_likelihood_mean"] = gaussian_circular
        bundle.loc[ordered_positions, "gaussian_circular_likelihood_rank"] = rank_descending(
            gaussian_circular
        )
        bundle.loc[ordered_positions, "circular_rotation"] = rotation

    bundle["student_t_circular_likelihood_rank"] = bundle[
        "student_t_circular_likelihood_rank"
    ].astype(np.int64)
    bundle["gaussian_circular_likelihood_rank"] = bundle[
        "gaussian_circular_likelihood_rank"
    ].astype(np.int64)
    bundle["circular_rotation"] = bundle["circular_rotation"].astype(np.int64)
    score_columns = [
        "student_t_likelihood_mean",
        "student_t_circular_likelihood_mean",
        "gaussian_likelihood_mean",
        "gaussian_circular_likelihood_mean",
    ]
    if not np.isfinite(bundle[score_columns].to_numpy(np.float64)).all():
        raise ValueError("target-free score bundle must be finite")
    rank_columns = [
        "student_t_likelihood_rank",
        "student_t_circular_likelihood_rank",
        "gaussian_likelihood_rank",
        "gaussian_circular_likelihood_rank",
    ]
    valid_rank = bundle[rank_columns].apply(
        lambda values: values.between(1, len(EXPECTED_SHIFTS)).all()
    )
    if not bool(valid_rank.all()):
        raise ValueError("target-free score ranks must cover 1..13")
    parity = float(
        np.mean(
            bundle["gaussian_likelihood_rank"].to_numpy(np.int64)
            == gaussian["likelihood_rank"].to_numpy(np.int64)
        )
    )
    control = {
        "saved_gaussian_rank_parity": parity,
        "score_finite_coverage": float(
            np.isfinite(bundle[score_columns].to_numpy(np.float64)).mean()
        ),
        "row_identity_coverage": 1.0,
        "circular_rotation_min": int(bundle["circular_rotation"].min()),
        "circular_rotation_max": int(bundle["circular_rotation"].max()),
    }
    return bundle.sort_values(keys, kind="mergesort").reset_index(drop=True), control


# %% [markdown]
# ## 6. Truth-only block labels and persistent-offset readout


# %%
def persistent_offset_episodes(
    signed_base_error: np.ndarray,
    row_idx: np.ndarray,
    *,
    threshold_ft: float,
    minimum_consecutive_rows: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    error = np.asarray(signed_base_error, dtype=np.float64)
    indices = np.asarray(row_idx, dtype=np.int64)
    if error.ndim != 1 or len(error) != len(indices):
        raise ValueError("persistent-offset inputs must be aligned one-dimensional arrays")
    bad = np.abs(error) > float(threshold_ft)
    padded = np.concatenate([[False], bad, [False]])
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    mask = np.zeros(len(error), dtype=bool)
    episodes: list[dict[str, Any]] = []
    for start, end in zip(starts, ends, strict=True):
        if end - start < int(minimum_consecutive_rows):
            continue
        mask[start:end] = True
        segment = error[start:end]
        episodes.append(
            {
                "episode_start_row_idx": int(indices[start]),
                "episode_end_row_idx": int(indices[end - 1]),
                "episode_rows": int(end - start),
                "median_signed_base_error_ft": float(np.median(segment)),
                "peak_abs_base_error_ft": float(np.max(np.abs(segment))),
            }
        )
    return mask, episodes


def sign_match(selected_shift: float, nearest_shift: float) -> bool:
    return bool(np.sign(float(selected_shift)) == np.sign(float(nearest_shift)))


def build_truth_readout(
    target_free_score_bundle: pd.DataFrame,
    oof_safe: pd.DataFrame,
    truth: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(truth) != len(oof_safe):
        raise ValueError("truth and safe OOF row counts must match before attachment")
    merged = oof_safe.merge(truth, on=["well_id", "row_idx"], how="left", validate="one_to_one")
    if len(merged) != len(oof_safe) or merged["tvt_true"].isna().any():
        raise ValueError("truth attachment failed row identity coverage")
    merged = merged.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)
    shifts = np.asarray(get_nested(config, "audit.shift_bank_ft"), dtype=np.float64)
    block_rows = int(get_nested(config, "audit.block_rows"))
    persistent_spec = get_nested(config, "audit.persistent_offset") or {}
    maximum_quantization_error = float(
        get_nested(config, "audit.coverage.maximum_quantization_error_ft")
    )
    readout_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for well, well_frame in merged.groupby("well_id", sort=True):
        well_frame = well_frame.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
        fold = int(well_frame["fold"].iloc[0])
        row_index = well_frame["row_idx"].to_numpy(np.int64)
        base_error = well_frame["tvt_geop"].to_numpy(np.float64) - well_frame["tvt_true"].to_numpy(
            np.float64
        )
        persistent_mask, well_episodes = persistent_offset_episodes(
            base_error,
            row_index,
            threshold_ft=float(persistent_spec["error_threshold_ft"]),
            minimum_consecutive_rows=int(persistent_spec["minimum_consecutive_rows"]),
        )
        for episode_id, row in enumerate(well_episodes):
            episode_rows.append(
                {"well_id": str(well), "fold": fold, "episode_id": episode_id, **row}
            )
        block_id = well_frame["suffix_offset"].to_numpy(np.int64) // block_rows
        for block in np.unique(block_id):
            mask = block_id == block
            block_frame = well_frame.loc[mask]
            block_scores = target_free_score_bundle.loc[
                (target_free_score_bundle["well_id"].astype(str) == str(well))
                & (target_free_score_bundle["block_id"] == int(block))
            ].sort_values("shift_slot", kind="mergesort")
            if len(block_scores) != len(shifts) or not np.array_equal(
                block_scores["shift_ft"].to_numpy(np.float64), shifts
            ):
                raise ValueError(f"target-free score bank misalignment for {well} block {block}")
            true_tvt = block_frame["tvt_true"].to_numpy(np.float64)
            geop = block_frame["tvt_geop"].to_numpy(np.float64)
            errors = geop[:, None] + shifts[None, :] - true_tvt[:, None]
            candidate_rmse = np.sqrt(np.mean(errors**2, axis=0))
            nearest_slot = int(np.argmin(candidate_rmse))
            continuous_optimal_shift = float(np.mean(true_tvt - geop))
            nearest_shift = float(shifts[nearest_slot])
            nearest_rmse = float(candidate_rmse[nearest_slot])
            base_rmse = float(np.sqrt(np.mean((geop - true_tvt) ** 2)))
            base_row_positions = np.flatnonzero(mask)
            score_meta = block_scores.iloc[0]
            row: dict[str, Any] = {
                "well_id": str(well),
                "fold": fold,
                "block_id": int(block),
                "block_start_row_idx": int(block_frame["row_idx"].iloc[0]),
                "block_end_row_idx": int(block_frame["row_idx"].iloc[-1]),
                "block_row_count": len(block_frame),
                "md_since_min_ft": float(score_meta["md_since_min_ft"]),
                "md_since_max_ft": float(score_meta["md_since_max_ft"]),
                "md_since_mid_ft": float(score_meta["md_since_mid_ft"]),
                "observed_gr_share": float(score_meta["observed_gr_share"]),
                "continuous_optimal_shift_ft": continuous_optimal_shift,
                "nearest_shift_ft": nearest_shift,
                "nearest_shift_slot": nearest_slot,
                "base_rmse": base_rmse,
                "nearest_shift_rmse": nearest_rmse,
                "oracle_shift_gain_rmse": float(base_rmse - nearest_rmse),
                "bank_range_covered": bool(
                    shifts.min() <= continuous_optimal_shift <= shifts.max()
                ),
                "nearest_shift_quantization_error_ft": float(
                    abs(nearest_shift - continuous_optimal_shift)
                ),
                "quantization_covered": bool(
                    abs(nearest_shift - continuous_optimal_shift) <= maximum_quantization_error
                ),
                "persistent_offset_share": float(persistent_mask[base_row_positions].mean()),
                "persistent_offset_block": bool(persistent_mask[base_row_positions].any()),
                "extreme_abs_z_ge_3_count": int(
                    block_scores["extreme_abs_z_ge_3_count"].iloc[nearest_slot]
                ),
                "extreme_abs_z_ge_3_share": float(
                    block_scores["extreme_abs_z_ge_3_share"].iloc[nearest_slot]
                ),
                "extreme_abs_z_ge_3_block": bool(
                    int(block_scores["extreme_abs_z_ge_3_count"].iloc[nearest_slot])
                    >= int(
                        get_nested(
                            config,
                            "validation.stage_0_pass_requires_all.extreme_residual."
                            "minimum_extreme_rows_per_block",
                        )
                    )
                ),
            }
            for family in ("student_t", "gaussian"):
                rank_values = block_scores[f"{family}_likelihood_rank"].to_numpy(np.int64)
                circular_rank_values = block_scores[f"{family}_circular_likelihood_rank"].to_numpy(
                    np.int64
                )
                likelihood = block_scores[f"{family}_likelihood_mean"].to_numpy(np.float64)
                real_rank = int(rank_values[nearest_slot])
                circular_rank = int(circular_rank_values[nearest_slot])
                top1_slot = int(np.argmin(rank_values))
                circular_top1_slot = int(np.argmin(circular_rank_values))
                ordered_likelihood = np.sort(likelihood)[::-1]
                other = np.delete(likelihood, nearest_slot)
                top1_rmse = float(candidate_rmse[top1_slot])
                row.update(
                    {
                        f"{family}_nearest_shift_rank": real_rank,
                        f"{family}_nearest_shift_circular_rank": circular_rank,
                        f"{family}_top1_hit": bool(real_rank == 1),
                        f"{family}_top3_hit": bool(real_rank <= 3),
                        f"{family}_mrr": float(1.0 / real_rank),
                        f"{family}_circular_top1_hit": bool(circular_rank == 1),
                        f"{family}_circular_top3_hit": bool(circular_rank <= 3),
                        f"{family}_circular_mrr": float(1.0 / circular_rank),
                        f"{family}_top1_shift_ft": float(shifts[top1_slot]),
                        f"{family}_circular_top1_shift_ft": float(shifts[circular_top1_slot]),
                        f"{family}_sign_match": sign_match(float(shifts[top1_slot]), nearest_shift),
                        f"{family}_circular_sign_match": sign_match(
                            float(shifts[circular_top1_slot]), nearest_shift
                        ),
                        f"{family}_likelihood_top1_margin": float(
                            ordered_likelihood[0] - ordered_likelihood[1]
                        ),
                        f"{family}_truth_candidate_margin": float(
                            likelihood[nearest_slot] - np.max(other)
                        ),
                        f"{family}_top1_shift_rmse": top1_rmse,
                        f"{family}_top1_regret_rmse": float(top1_rmse - nearest_rmse),
                    }
                )
            readout_rows.append(row)
    readout = pd.DataFrame(readout_rows).sort_values(["well_id", "block_id"], kind="mergesort")
    episodes = pd.DataFrame(episode_rows)
    return readout.reset_index(drop=True), episodes.reset_index(drop=True)


# %% [markdown]
# ## 7. Fold, stress, extreme-residual metrics and fixed gate


# %%
def readout_metric_row(frame: pd.DataFrame, *, scope: str) -> dict[str, Any]:
    if frame.empty:
        raise ValueError(f"scope {scope} selected zero blocks")
    row: dict[str, Any] = {
        "scope": scope,
        "blocks": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "bank_range_coverage": float(frame["bank_range_covered"].mean()),
        "quantization_coverage": float(frame["quantization_covered"].mean()),
        "oracle_shift_gain_rmse_mean": float(frame["oracle_shift_gain_rmse"].mean()),
        "extreme_abs_z_ge_3_block_share": float(frame["extreme_abs_z_ge_3_block"].mean()),
    }
    for family in ("student_t", "gaussian"):
        top1 = float(frame[f"{family}_top1_hit"].mean())
        top3 = float(frame[f"{family}_top3_hit"].mean())
        mrr = float(frame[f"{family}_mrr"].mean())
        circular_top1 = float(frame[f"{family}_circular_top1_hit"].mean())
        circular_top3 = float(frame[f"{family}_circular_top3_hit"].mean())
        circular_mrr = float(frame[f"{family}_circular_mrr"].mean())
        row.update(
            {
                f"{family}_top1_rate": top1,
                f"{family}_top3_rate": top3,
                f"{family}_mrr": mrr,
                f"{family}_mean_rank": float(frame[f"{family}_nearest_shift_rank"].mean()),
                f"{family}_sign_accuracy": float(frame[f"{family}_sign_match"].mean()),
                f"{family}_circular_top1_rate": circular_top1,
                f"{family}_circular_top3_rate": circular_top3,
                f"{family}_circular_mrr": circular_mrr,
                f"{family}_top1_gap_vs_circular": top1 - circular_top1,
                f"{family}_top3_gap_vs_circular": top3 - circular_top3,
                f"{family}_mrr_gap_vs_circular": mrr - circular_mrr,
                f"{family}_top1_regret_rmse_mean": float(
                    frame[f"{family}_top1_regret_rmse"].mean()
                ),
                f"{family}_top1_regret_rmse_p90": float(
                    frame[f"{family}_top1_regret_rmse"].quantile(0.90)
                ),
                f"{family}_likelihood_top1_margin_mean": float(
                    frame[f"{family}_likelihood_top1_margin"].mean()
                ),
                f"{family}_truth_candidate_margin_mean": float(
                    frame[f"{family}_truth_candidate_margin"].mean()
                ),
            }
        )
    row.update(
        {
            "student_t_minus_gaussian_top1_rate": (
                row["student_t_top1_rate"] - row["gaussian_top1_rate"]
            ),
            "student_t_minus_gaussian_top3_rate": (
                row["student_t_top3_rate"] - row["gaussian_top3_rate"]
            ),
            "student_t_minus_gaussian_mrr": (row["student_t_mrr"] - row["gaussian_mrr"]),
            "student_t_minus_gaussian_top1_regret_rmse_mean": (
                row["student_t_top1_regret_rmse_mean"] - row["gaussian_top1_regret_rmse_mean"]
            ),
            "student_t_minus_gaussian_top3_gap_vs_circular": (
                row["student_t_top3_gap_vs_circular"] - row["gaussian_top3_gap_vs_circular"]
            ),
            "student_t_minus_gaussian_mrr_gap_vs_circular": (
                row["student_t_mrr_gap_vs_circular"] - row["gaussian_mrr_gap_vs_circular"]
            ),
        }
    )
    return row


def build_scope_metrics(
    readout: pd.DataFrame,
    hidden_assignments: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scope_rows = [readout_metric_row(readout, scope="overall")]
    near_limit = float(get_nested(config, "audit.scopes.near_max_md_since_ft"))
    long_limit = float(get_nested(config, "audit.scopes.long_tail_min_md_since_ft"))
    predefined = {
        "near": readout["md_since_mid_ft"] < near_limit,
        "long_tail_1000_plus": readout["md_since_mid_ft"] >= long_limit,
        "persistent_offset": readout["persistent_offset_block"].astype(bool),
        "extreme_abs_z_ge_3": readout["extreme_abs_z_ge_3_block"].astype(bool),
    }
    for name, mask in predefined.items():
        if bool(mask.any()):
            scope_rows.append(readout_metric_row(readout.loc[mask], scope=name))
    if not hidden_assignments.empty:
        role_columns = get_nested(config, "data.hidden_like.role_columns") or {}
        role_by_well = hidden_assignments.set_index("well_id")
        for scope_name, role_column in role_columns.items():
            valid_wells = set(
                role_by_well.index[role_by_well[str(role_column)].astype(str) == "valid"].astype(
                    str
                )
            )
            part = readout.loc[readout["well_id"].astype(str).isin(valid_wells)]
            scope_rows.append(readout_metric_row(part, scope=str(scope_name)))
    fold_rows = []
    for fold, part in readout.groupby("fold", sort=True):
        row = readout_metric_row(part, scope=f"fold_{int(fold)}")
        row["fold"] = int(fold)
        fold_rows.append(row)
    return pd.DataFrame(scope_rows), pd.DataFrame(fold_rows)


def build_shift_metrics(readout: pd.DataFrame, shifts: list[float]) -> pd.DataFrame:
    rows = []
    for shift in shifts:
        nearest = readout.loc[np.isclose(readout["nearest_shift_ft"], float(shift))]
        student_selected = readout.loc[np.isclose(readout["student_t_top1_shift_ft"], float(shift))]
        gaussian_selected = readout.loc[np.isclose(readout["gaussian_top1_shift_ft"], float(shift))]
        rows.append(
            {
                "shift_ft": float(shift),
                "truth_nearest_blocks": len(nearest),
                "truth_nearest_share": float(len(nearest) / len(readout)),
                "student_t_top1_blocks": len(student_selected),
                "student_t_top1_share": float(len(student_selected) / len(readout)),
                "gaussian_top1_blocks": len(gaussian_selected),
                "gaussian_top1_share": float(len(gaussian_selected) / len(readout)),
                "student_t_top1_rate_when_truth_nearest": float(
                    nearest["student_t_top1_hit"].mean()
                )
                if len(nearest)
                else np.nan,
                "student_t_top3_rate_when_truth_nearest": float(
                    nearest["student_t_top3_hit"].mean()
                )
                if len(nearest)
                else np.nan,
                "gaussian_top3_rate_when_truth_nearest": float(nearest["gaussian_top3_hit"].mean())
                if len(nearest)
                else np.nan,
                "student_t_mean_rank_when_truth_nearest": float(
                    nearest["student_t_nearest_shift_rank"].mean()
                )
                if len(nearest)
                else np.nan,
                "gaussian_mean_rank_when_truth_nearest": float(
                    nearest["gaussian_nearest_shift_rank"].mean()
                )
                if len(nearest)
                else np.nan,
                "mean_quantization_error_ft": float(
                    nearest["nearest_shift_quantization_error_ft"].mean()
                )
                if len(nearest)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_by_well_metrics(readout: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for well, part in readout.groupby("well_id", sort=True):
        row = readout_metric_row(part, scope=str(well))
        row["well_id"] = str(well)
        row["fold"] = int(part["fold"].iloc[0])
        row["persistent_offset_blocks"] = int(part["persistent_offset_block"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_guard(
    technical_control: dict[str, Any],
    scope_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    guards = get_nested(config, "validation.stage_0_pass_requires_all") or {}
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    actual_folds = sorted(int(value) for value in fold_metrics["fold"].unique())
    scopes = scope_metrics.set_index("scope")
    pooled = scopes.loc["overall"]
    improved_folds_mrr = int((fold_metrics["student_t_minus_gaussian_mrr"] > 0.0).sum())
    improved_folds_top3 = int((fold_metrics["student_t_minus_gaussian_top3_rate"] > 0.0).sum())
    stress_scopes = [str(value) for value in guards["stress_scopes"]]
    missing_scopes = sorted(set(stress_scopes).difference(scopes.index))
    if missing_scopes:
        raise ValueError(f"required stress scopes are missing: {missing_scopes}")
    stress_mrr = bool((scopes.loc[stress_scopes, "student_t_minus_gaussian_mrr"] >= 0.0).all())
    stress_top3 = bool(
        (scopes.loc[stress_scopes, "student_t_minus_gaussian_top3_rate"] >= 0.0).all()
    )
    extreme_scope_present = "extreme_abs_z_ge_3" in scopes.index
    extreme = scopes.loc["extreme_abs_z_ge_3"] if extreme_scope_present else None
    pooled_mrr_pass = bool(
        float(pooled["student_t_minus_gaussian_mrr"]) >= float(guards["minimum_pooled_mrr_gain"])
    )
    pooled_top3_pass = bool(
        float(pooled["student_t_minus_gaussian_top3_rate"])
        >= float(guards["minimum_pooled_top3_gain"])
    )
    extreme_top3_pass = bool(
        extreme_scope_present and float(extreme["student_t_minus_gaussian_top3_rate"]) > 0.0
    )
    extreme_regret_pass = bool(
        extreme_scope_present
        and float(extreme["student_t_minus_gaussian_top1_regret_rmse_mean"]) < 0.0
    )
    checks = {
        "expected_folds": actual_folds == expected_folds,
        "score_finite_coverage": float(technical_control["score_finite_coverage"])
        >= float(guards["required_score_finite_coverage"]),
        "row_identity_coverage": float(technical_control["row_identity_coverage"])
        >= float(guards["required_row_identity_coverage"]),
        "saved_gaussian_rank_parity": float(technical_control["saved_gaussian_rank_parity"])
        >= float(guards["required_control_rank_parity"]),
        "pooled_mrr_gain": pooled_mrr_pass,
        "pooled_top3_gain": pooled_top3_pass,
        "mrr_improved_folds": improved_folds_mrr >= int(guards["minimum_improved_folds_mrr"]),
        "top3_improved_folds": improved_folds_top3 >= int(guards["minimum_improved_folds_top3"]),
        "stress_mrr_non_regression": stress_mrr,
        "stress_top3_non_regression": stress_top3,
        "mrr_gap_vs_circular_not_worse": float(
            pooled["student_t_minus_gaussian_mrr_gap_vs_circular"]
        )
        >= 0.0,
        "top3_gap_vs_circular_not_worse": float(
            pooled["student_t_minus_gaussian_top3_gap_vs_circular"]
        )
        >= 0.0,
        "extreme_residual_top3_improvement": extreme_top3_pass,
        "extreme_residual_regret_improvement": extreme_regret_pass,
    }
    flattening_signal = bool(
        float(pooled["student_t_likelihood_top1_margin_mean"])
        < float(pooled["gaussian_likelihood_top1_margin_mean"])
    )
    exp344_pattern = bool(
        extreme_top3_pass
        and extreme_regret_pass
        and not (pooled_mrr_pass and pooled_top3_pass)
        and flattening_signal
    )
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "actual_folds": actual_folds,
        "technical_control": technical_control,
        "improved_folds": {"mrr": improved_folds_mrr, "top3": improved_folds_top3},
        "stress_scopes": stress_scopes,
        "extreme_residual": {
            "scope_present": extreme_scope_present,
            "blocks": int(extreme["blocks"]) if extreme_scope_present else 0,
            "top3_gain": (
                float(extreme["student_t_minus_gaussian_top3_rate"])
                if extreme_scope_present
                else None
            ),
            "regret_delta_ft": (
                float(extreme["student_t_minus_gaussian_top1_regret_rmse_mean"])
                if extreme_scope_present
                else None
            ),
        },
        "flattening_signal": flattening_signal,
        "exp344_dependency_pattern_matched": exp344_pattern,
        "stage_1_eligible": bool(all(checks.values())),
        "decision": (
            "stage_0_passed_stage_1_requires_separate_approval"
            if all(checks.values())
            else (
                "stage_0_failed_exp344_dependency_pattern_matched"
                if exp344_pattern
                else "stage_0_failed_close_without_rescue"
            )
        ),
    }


# %% [markdown]
# ## 8. Stage 0 Kaggle CPU orchestration and artifact guards


# %%
def run_stage_0_experiment(config: dict[str, Any]) -> dict[str, Any]:
    if not KAGGLE_WORKING_ROOT.exists() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError(
            "Full exp342 Stage 0 must run on Kaggle. EXPERIMENT_ALLOW_LOCAL=1 is reserved "
            "for an explicitly approved local smoke run."
        )
    validate_scientific_contract(config, require_run_approval=True)
    started = time.time()
    gaussian_control, gaussian_evidence = load_exp280_gaussian_control(config)
    safe_oof, exp226_path, exp226_manifest = load_exp226_safe(config)
    raw_dir = train_data_dir(config)
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    raw_wells = sorted(
        path.name.replace("__horizontal_well.csv", "")
        for path in raw_dir.glob("*__horizontal_well.csv")
    )
    if len(raw_wells) != expected_wells or set(raw_wells) != set(safe_oof["well_id"].unique()):
        raise ValueError("raw train and exp226 well sets do not match")

    student_score_parts: list[pd.DataFrame] = []
    well_manifest_rows: list[dict[str, Any]] = []
    progress_every = 25
    for index, well in enumerate(raw_wells, start=1):
        horizontal_path = raw_dir / f"{well}__horizontal_well.csv"
        typewell_path = raw_dir / f"{well}__typewell.csv"
        if not typewell_path.exists():
            raise FileNotFoundError(typewell_path)
        horizontal_safe = load_horizontal_without_truth(horizontal_path)
        typewell = pd.read_csv(typewell_path)
        well_scores, well_manifest = score_well_student_t_target_free(
            safe_oof.loc[safe_oof["well_id"] == well], horizontal_safe, typewell, config
        )
        well_manifest.update(
            {
                "horizontal_path": str(horizontal_path),
                "horizontal_raw_sha256": sha256_path(horizontal_path),
                "typewell_path": str(typewell_path),
                "typewell_raw_sha256": sha256_path(typewell_path),
            }
        )
        student_score_parts.append(well_scores)
        well_manifest_rows.append(well_manifest)
        if index % progress_every == 0 or index == len(raw_wells):
            print(f"Student-t target-free scoring wells={index}/{len(raw_wells)}")

    student_scores = pd.concat(student_score_parts, ignore_index=True).sort_values(
        ["well_id", "block_id", "shift_slot"], kind="mergesort"
    )
    target_free_bundle, technical_control = build_target_free_score_bundle(
        student_scores, gaussian_control, config
    )
    target_free_score_content_sha = dataframe_content_sha(target_free_bundle)
    if not target_free_score_content_sha:
        raise RuntimeError("failed to freeze target-free Student-t/control bundle SHA")
    artifacts = artifact_dir()
    score_contract = {
        "experiment": EXPERIMENT_NAME,
        "truth_attached": False,
        "stage": "stage_0",
        "shift_bank_ft": get_nested(config, "audit.shift_bank_ft"),
        "block_rows": get_nested(config, "audit.block_rows"),
        "block_policy": get_nested(config, "audit.block_policy"),
        "score_aggregation": get_nested(config, "audit.score_aggregation"),
        "tie_policy": get_nested(config, "audit.tie_policy"),
        "student_t_emission": get_nested(config, "audit.student_t_emission"),
        "gaussian_control": get_nested(config, "model.stage_0.gaussian_control"),
        "gaussian_control_content_sha256": get_nested(
            config, "data.exp280_source.score_content_sha256"
        ),
        "circular_control": get_nested(config, "audit.circular_control"),
        "extreme_residual": get_nested(
            config,
            "validation.stage_0_pass_requires_all.extreme_residual",
        ),
        "target_free_score_content_sha256": target_free_score_content_sha,
    }
    score_contract["scientific_contract_sha256"] = mapping_sha256(score_contract)
    score_contract_path = artifacts / f"{OUTPUT_PREFIX}_score_contract.json"
    write_json(score_contract_path, score_contract)
    score_artifact = write_csv_gzip(
        target_free_bundle,
        artifacts / f"{OUTPUT_PREFIX}_target_free_score_bundle.csv.gz",
    )

    # Truth is first read here, after Student-t and saved-control scores are frozen.
    truth = load_exp226_truth(
        exp226_path,
        config,
        frozen_score_content_sha256=target_free_score_content_sha,
    )
    readout, episodes = build_truth_readout(target_free_bundle, safe_oof, truth, config)
    hidden_assignments, hidden_manifest = load_hidden_like_assignments(config)
    scope_metrics, fold_metrics = build_scope_metrics(readout, hidden_assignments, config)
    shift_metrics = build_shift_metrics(
        readout, [float(value) for value in get_nested(config, "audit.shift_bank_ft")]
    )
    by_well = build_by_well_metrics(readout)
    well_manifest = pd.DataFrame(well_manifest_rows).sort_values("well_id", kind="mergesort")
    guard = evaluate_guard(technical_control, scope_metrics, fold_metrics, config)
    gate_path = artifacts / f"{OUTPUT_PREFIX}_gate.json"
    write_json(gate_path, guard)

    readout_artifact = write_csv_gzip(
        readout,
        artifacts / f"{OUTPUT_PREFIX}_block_readout.csv.gz",
    )
    file_frames = {
        "scope_metrics": scope_metrics,
        "fold_metrics": fold_metrics,
        "shift_metrics": shift_metrics,
        "by_well_metrics": by_well,
        "persistent_offset_episodes": episodes,
        "well_manifest": well_manifest,
    }
    output_paths: dict[str, Path] = {}
    for name, frame in file_frames.items():
        path = artifacts / f"{OUTPUT_PREFIX}_{name}.csv"
        frame.to_csv(path, index=False)
        output_paths[name] = path

    input_manifest = pd.DataFrame(
        [
            *gaussian_evidence,
            exp226_manifest,
            hidden_manifest,
            {
                "name": "raw_train_well_files",
                "path": str(raw_dir),
                "rows": int(well_manifest["horizontal_rows"].sum()),
                "wells": len(well_manifest),
                "raw_sha256": dataframe_content_sha(
                    well_manifest,
                    ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"],
                ),
            },
        ]
    )
    input_manifest_path = artifacts / f"{OUTPUT_PREFIX}_input_manifest.csv"
    input_manifest.to_csv(input_manifest_path, index=False)

    overall = scope_metrics.loc[scope_metrics["scope"] == "overall"].iloc[0].to_dict()
    hashed_outputs = {
        **output_paths,
        "input_manifest": input_manifest_path,
        "gate": gate_path,
        "score_contract": score_contract_path,
    }
    output_sha = {name: sha256_path(path) for name, path in hashed_outputs.items()}
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "stage_0_completed_guard_passed"
        if guard["passed"]
        else "stage_0_completed_guard_failed",
        "route": get_nested(config, "experiment.route"),
        "runtime_seconds": time.time() - started,
        "rows": len(safe_oof),
        "wells": int(safe_oof["well_id"].nunique()),
        "blocks": len(readout),
        "shift_candidates": len(get_nested(config, "audit.shift_bank_ft")),
        "scientific_scores": 1,
        "control_scores": 1,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "overall": overall,
        "guard": guard,
        "truth_attachment": {
            "stage": "after_student_t_and_gaussian_control_bundle_frozen",
            "target_free_score_content_sha256": target_free_score_content_sha,
        },
        "input_manifest": input_manifest.to_dict(orient="records"),
        "artifacts": {
            "score_contract": str(score_contract_path),
            "target_free_score_bundle": score_artifact,
            "block_readout": readout_artifact,
            "gate": str(gate_path),
            "file_sha256": output_sha,
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "decision": guard["decision"],
        "stage_1_implemented": False,
        "stage_1_run": False,
        "inference_run": False,
        "submission_created": False,
    }
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"
    write_json(summary_path, summary)
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": summary["status"],
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "metric": get_nested(config, "validation.metric"),
        "diagnostic": {
            "overall": overall,
            "guard": guard,
            "target_free_score_content_sha256": target_free_score_content_sha,
        },
        "notes": (
            "Stage 0 only. No HMM path, model, prediction, inference, or submission is produced."
        ),
    }
    write_json(metrics_output_path(), metrics)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 9. Stage 1 exact-HMM kernel and Student-t path generation


# %%
STAGE_1_CANDIDATES = (
    "exp226_pred",
    "exp263_fixed",
    "gaussian_residual_offset_hmm",
    "student_t_residual_offset_hmm",
)


def list_stage_1_wells(data_dir: str | Path) -> list[str]:
    root = Path(data_dir)
    wells: list[str] = []
    for path in sorted(root.glob("*__horizontal_well.csv")):
        well = path.stem.replace("__horizontal_well", "")
        if (root / f"{well}__typewell.csv").exists():
            wells.append(well)
    return wells


def load_stage_1_well(
    well: str, data_dir: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(data_dir)
    horizontal = pd.read_csv(
        root / f"{well}__horizontal_well.csv",
        usecols=lambda column: column != "TVT",
    )
    typewell = (
        pd.read_csv(root / f"{well}__typewell.csv")
        .sort_values("TVT", kind="mergesort")
        .reset_index(drop=True)
    )
    if "TVT" in horizontal.columns:
        raise AssertionError("Stage 1 candidate generation unexpectedly contains true TVT")
    return horizontal, typewell


def robust_initial_rate(
    known_prefix: pd.DataFrame,
    window_rows: int = 30,
    *,
    min_valid_steps: int = 3,
    fallback_rate: float = 0.0,
) -> tuple[float, int, int]:
    tail = known_prefix.tail(int(window_rows))
    tvt = pd.to_numeric(tail["TVT_input"], errors="coerce").to_numpy(np.float64)
    z = pd.to_numeric(tail["Z"], errors="coerce").to_numpy(np.float64)
    md = pd.to_numeric(tail["MD"], errors="coerce").to_numpy(np.float64)
    dtvt, dz, dmd = np.diff(tvt), np.diff(z), np.diff(md)
    valid = np.isfinite(dtvt) & np.isfinite(dz) & np.isfinite(dmd) & (dmd > 0.0)
    valid_steps = int(valid.sum())
    if valid_steps < int(min_valid_steps):
        return float(fallback_rate), int(len(tail)), valid_steps
    rate = float(np.median((dtvt[valid] + dz[valid]) / dmd[valid]))
    if not np.isfinite(rate):
        rate = float(fallback_rate)
    return rate, int(len(tail)), valid_steps


def prefix_stats(
    horizontal: pd.DataFrame,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    tail_n: int = 30,
) -> tuple[float, float, float, float, int, int]:
    known = horizontal.loc[horizontal["TVT_input"].notna()]
    known_gr = known["GR"].to_numpy(np.float64)
    known_tvt = known["TVT_input"].to_numpy(np.float64)
    typewell_at_known = np.interp(known_tvt, typewell_tvt, typewell_gr)
    valid = np.isfinite(known_gr) & np.isfinite(typewell_at_known)
    if valid.sum() >= 20 and np.std(typewell_at_known[valid]) > 1.0e-6:
        cal_a, cal_b = np.polyfit(typewell_at_known[valid], known_gr[valid], 1)
    elif valid.any():
        cal_a = 1.0
        cal_b = float(np.nanmean(known_gr) - np.nanmean(typewell_at_known))
    else:
        cal_a, cal_b = 1.0, 0.0
    residual = known_gr[valid] - (cal_a * typewell_at_known[valid] + cal_b)
    if valid.sum() > 20:
        sigma = float(
            np.clip(
                1.4826 * np.median(np.abs(residual - np.median(residual))),
                8.0,
                60.0,
            )
        )
    else:
        sigma = 30.0
    init_rate, effective_rows, valid_steps = robust_initial_rate(known, tail_n)
    return float(cal_a), float(cal_b), sigma, init_rate, effective_rows, valid_steps


def prepare_student_t_hmm_inputs(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    geop_tvt: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    if "TVT" in horizontal.columns:
        raise ValueError("Stage 1 HMM preparation forbids unknown-suffix true TVT")
    required_horizontal = {"MD", "Z", "GR", "TVT_input"}
    if not required_horizontal.issubset(horizontal.columns):
        raise ValueError(
            f"horizontal missing {sorted(required_horizontal - set(horizontal.columns))}"
        )
    if not {"TVT", "GR"}.issubset(typewell.columns):
        raise ValueError("typewell must contain TVT and GR")
    hmm = get_nested(config, "model.hmm") or {}
    typewell_tvt = pd.to_numeric(typewell["TVT"], errors="raise").to_numpy(np.float64)
    typewell_gr = (
        pd.to_numeric(typewell["GR"], errors="coerce")
        .ffill()
        .bfill()
        .to_numpy(np.float64)
    )
    if len(typewell_tvt) < 2 or not np.isfinite(typewell_tvt).all():
        raise ValueError("typewell TVT must contain at least two finite rows")
    known = horizontal.loc[horizontal["TVT_input"].notna()]
    eval_rows = horizontal.loc[horizontal["TVT_input"].isna()]
    if len(known) < 4 or len(eval_rows) == 0:
        raise ValueError("well requires at least four known rows and one evaluation row")
    geop = np.asarray(geop_tvt, dtype=np.float64)
    if len(geop) != len(eval_rows) or not np.isfinite(geop).all():
        raise ValueError("exp226 tvt_geop must be finite and align to every evaluation row")

    cal_a, cal_b, robust_sigma, init_rate, rate_rows, valid_steps = prefix_stats(
        horizontal, typewell_tvt, typewell_gr, tail_n=30
    )
    if hmm["sigma_mode"] == "std":
        known_tvt = known["TVT_input"].to_numpy(np.float64)
        typewell_at_known = np.interp(known_tvt, typewell_tvt, typewell_gr)
        residual = known["GR"].fillna(0).to_numpy(np.float64) - typewell_at_known
        gr_sigma = float(np.clip(np.nanstd(residual), 10.0, 60.0))
        cal_a_use, cal_b_use = 1.0, 0.0
    else:
        gr_sigma = robust_sigma
        cal_a_use, cal_b_use = cal_a, cal_b
    if not np.isfinite(gr_sigma):
        raise ValueError("Stage 1 known-prefix GR sigma must be finite")

    last = known.iloc[-1]
    step = float(hmm["step"])
    delta_min = float(hmm["delta_min_ft"])
    delta_max = float(hmm["delta_max_ft"])
    if not delta_min < 0.0 < delta_max:
        raise ValueError("residual-offset grid must straddle zero")
    grid = np.arange(delta_min, delta_max + 0.5 * step, step, dtype=np.float64)
    if len(grid) < 3:
        raise ValueError("residual-offset grid is too small")

    md = eval_rows["MD"].to_numpy(np.float64)
    gr_fill = float(np.nanmean(typewell_gr))
    gr = (
        horizontal["GR"]
        .interpolate(limit_direction="both")
        .fillna(gr_fill)
        .to_numpy(np.float64)[eval_rows.index]
    )
    dm = np.maximum(np.diff(np.concatenate([[float(last["MD"])], md])), 1.0)
    absolute_tvt_states = geop[:, None] + grid[None, :]
    gr_grid = (
        cal_a_use * np.interp(absolute_tvt_states, typewell_tvt, typewell_gr)
        + cal_b_use
    )
    zscore = (gr[:, None] - gr_grid) / gr_sigma
    if hmm["emission"] != "student_t_df4":
        raise ValueError("exp342 Stage 1 fixes the Student-t df=4 GR emission")
    emission_ll = student_t_log_likelihood(
        zscore, float(hmm["student_t_degrees_of_freedom"])
    ).astype(np.float32)

    if hmm["rate_center"] != "zero":
        raise ValueError("exp342 offset-rate states must be centered at zero")
    rates = np.linspace(
        -float(hmm["rate_span"]),
        float(hmm["rate_span"]),
        int(hmm["n_rates"]),
        dtype=np.float64,
    )
    start_delta = float(hmm["start_delta_ft"])
    zero_quantization_error = float(np.min(np.abs(grid - start_delta)))
    native_typewell = (absolute_tvt_states >= float(typewell_tvt.min())) & (
        absolute_tvt_states <= float(typewell_tvt.max())
    )
    return {
        "emission_ll": emission_ll,
        "dm": dm,
        "grid": grid,
        "rates": rates,
        "start_p": float((start_delta - grid[0]) / step),
        "r0": float(hmm["initial_offset_rate"]),
        "eval_index": eval_rows.index.to_numpy(np.int64),
        "prefix_rows": int(len(known)),
        "prefix_sigma": gr_sigma,
        "prefix_ir_diagnostic_only": init_rate,
        "initial_rate_effective_rows": int(rate_rows),
        "initial_rate_valid_steps": int(valid_steps),
        "grid_min": float(grid[0]),
        "grid_max": float(grid[-1]),
        "delta_zero_quantization_error_ft": zero_quantization_error,
        "delta_grid_coverage_rows": int(len(geop)),
        "delta_grid_rows": int(len(geop)),
        "native_typewell_state_coverage": float(native_typewell.mean()),
        "emission_finite_coverage": float(np.isfinite(emission_ll).mean()),
    }


@njit(cache=True, nogil=True, parallel=True)
def _hmm2_fb_student_t(
    em,
    dm,
    sp,
    rates,
    sig_r,
    sig_p,
    start_p,
    start_sig,
    r0,
    r0_sig,
    lam,
    mom,
):
    t_count, p_count = em.shape
    r_count = len(rates)
    rate_step = rates[1] - rates[0]
    neg = np.float32(-1e18)
    alpha = np.full((t_count, p_count, r_count), neg, np.float32)
    prev = np.full((p_count, r_count), neg, np.float32)
    for p_i in range(p_count):
        dpos = (p_i - start_p) * sp
        lp0 = -0.5 * (dpos / start_sig) ** 2
        if lp0 < -60.0:
            continue
        for r_i in range(r_count):
            dr = (rates[r_i] - r0) / r0_sig
            prev[p_i, r_i] = np.float32(lp0 - 0.5 * dr * dr)
    tmp = np.empty((p_count, r_count), np.float32)
    cur = np.empty((p_count, r_count), np.float32)

    for t_i in range(t_count):
        sig_rate_step = sig_r * np.sqrt(dm[t_i])
        rate_var_cells = (sig_rate_step / rate_step) ** 2
        rate_log_kernel = np.empty((r_count, 3))
        for r_i in range(r_count):
            mean_rate_move = -(1.0 - mom) * rates[r_i] * dm[t_i] / rate_step
            p_plus = max(0.5 * (rate_var_cells + mean_rate_move), 1e-12)
            p_minus = max(0.5 * (rate_var_cells - mean_rate_move), 1e-12)
            total = p_plus + p_minus
            if total > 0.9:
                p_plus *= 0.9 / total
                p_minus *= 0.9 / total
            rate_log_kernel[r_i, 0] = np.log(p_minus)
            rate_log_kernel[r_i, 1] = np.log(1.0 - p_plus - p_minus)
            rate_log_kernel[r_i, 2] = np.log(p_plus)
        for p_i in prange(p_count):
            for r2 in range(r_count):
                best = neg
                k0 = max(r2 - 1, 0)
                k1 = min(r2 + 1, r_count - 1)
                for r_i in range(k0, k1 + 1):
                    value = prev[p_i, r_i] + rate_log_kernel[r_i, r2 - r_i + 1]
                    if value > best:
                        best = value
                if best > neg / 2:
                    total = 0.0
                    for r_i in range(k0, k1 + 1):
                        total += np.exp(
                            prev[p_i, r_i]
                            + rate_log_kernel[r_i, r2 - r_i + 1]
                            - best
                        )
                    tmp[p_i, r2] = np.float32(best + np.log(total))
                else:
                    tmp[p_i, r2] = neg

        sigma_position = max(sig_p, 0.35 * sp)
        for r2 in prange(r_count):
            mu = rates[r2] * dm[t_i]
            b0 = int(np.floor(mu / sp + 0.5))
            position_log_kernel = np.empty(5)
            for k_i in range(5):
                delta = (b0 - 2 + k_i) * sp - mu
                position_log_kernel[k_i] = -0.5 * (delta / sigma_position) ** 2
            kernel_max = np.max(position_log_kernel)
            log_norm = kernel_max + np.log(
                np.sum(np.exp(position_log_kernel - kernel_max))
            )
            position_log_kernel -= log_norm
            for p2 in range(p_count):
                best = neg
                for k_i in range(5):
                    p1 = p2 - (b0 - 2 + k_i)
                    if 0 <= p1 < p_count:
                        value = tmp[p1, r2] + position_log_kernel[k_i]
                        if value > best:
                            best = value
                if best > neg / 2:
                    total = 0.0
                    for k_i in range(5):
                        p1 = p2 - (b0 - 2 + k_i)
                        if 0 <= p1 < p_count:
                            total += np.exp(
                                tmp[p1, r2] + position_log_kernel[k_i] - best
                            )
                    cur[p2, r2] = np.float32(
                        best + np.log(total) + lam * em[t_i, p2]
                    )
                else:
                    cur[p2, r2] = neg
        for p_i in range(p_count):
            for r_i in range(r_count):
                alpha[t_i, p_i, r_i] = cur[p_i, r_i]
                prev[p_i, r_i] = cur[p_i, r_i]

    best = np.float32(neg)
    for p_i in range(p_count):
        for r_i in range(r_count):
            if alpha[t_count - 1, p_i, r_i] > best:
                best = alpha[t_count - 1, p_i, r_i]
    total = 0.0
    for p_i in range(p_count):
        for r_i in range(r_count):
            total += np.exp(alpha[t_count - 1, p_i, r_i] - best)
    loglik = float(best) + np.log(total)
    post_p = np.zeros((t_count, p_count))
    beta_next = np.zeros((p_count, r_count), np.float32)
    values = alpha[t_count - 1] + beta_next
    best = np.max(values)
    total = 0.0
    for p_i in range(p_count):
        acc = 0.0
        for r_i in range(r_count):
            acc += np.exp(values[p_i, r_i] - best)
        post_p[t_count - 1, p_i] = acc
        total += acc
    post_p[t_count - 1] /= total

    beta_cur = np.empty((p_count, r_count), np.float32)
    beta_tmp = np.empty((p_count, r_count), np.float32)
    for t_i in range(t_count - 1, 0, -1):
        sig_rate_step = sig_r * np.sqrt(dm[t_i])
        rate_var_cells = (sig_rate_step / rate_step) ** 2
        rate_log_kernel = np.empty((r_count, 3))
        for r_i in range(r_count):
            mean_rate_move = -(1.0 - mom) * rates[r_i] * dm[t_i] / rate_step
            p_plus = max(0.5 * (rate_var_cells + mean_rate_move), 1e-12)
            p_minus = max(0.5 * (rate_var_cells - mean_rate_move), 1e-12)
            total = p_plus + p_minus
            if total > 0.9:
                p_plus *= 0.9 / total
                p_minus *= 0.9 / total
            rate_log_kernel[r_i, 0] = np.log(p_minus)
            rate_log_kernel[r_i, 1] = np.log(1.0 - p_plus - p_minus)
            rate_log_kernel[r_i, 2] = np.log(p_plus)
        sigma_position = max(sig_p, 0.35 * sp)
        for r2 in prange(r_count):
            mu = rates[r2] * dm[t_i]
            b0 = int(np.floor(mu / sp + 0.5))
            position_log_kernel = np.empty(5)
            for k_i in range(5):
                delta = (b0 - 2 + k_i) * sp - mu
                position_log_kernel[k_i] = -0.5 * (delta / sigma_position) ** 2
            kernel_max = np.max(position_log_kernel)
            log_norm = kernel_max + np.log(
                np.sum(np.exp(position_log_kernel - kernel_max))
            )
            position_log_kernel -= log_norm
            for p1 in range(p_count):
                best = neg
                for k_i in range(5):
                    p2 = p1 + (b0 - 2 + k_i)
                    if 0 <= p2 < p_count:
                        value = (
                            position_log_kernel[k_i]
                            + lam * em[t_i, p2]
                            + beta_next[p2, r2]
                        )
                        if value > best:
                            best = value
                if best > neg / 2:
                    total = 0.0
                    for k_i in range(5):
                        p2 = p1 + (b0 - 2 + k_i)
                        if 0 <= p2 < p_count:
                            total += np.exp(
                                position_log_kernel[k_i]
                                + lam * em[t_i, p2]
                                + beta_next[p2, r2]
                                - best
                            )
                    beta_tmp[p1, r2] = np.float32(best + np.log(total))
                else:
                    beta_tmp[p1, r2] = neg
        for p_i in prange(p_count):
            for r_i in range(r_count):
                best = neg
                k0 = max(r_i - 1, 0)
                k1 = min(r_i + 1, r_count - 1)
                for r2 in range(k0, k1 + 1):
                    value = (
                        rate_log_kernel[r_i, r2 - r_i + 1] + beta_tmp[p_i, r2]
                    )
                    if value > best:
                        best = value
                if best > neg / 2:
                    total = 0.0
                    for r2 in range(k0, k1 + 1):
                        total += np.exp(
                            rate_log_kernel[r_i, r2 - r_i + 1]
                            + beta_tmp[p_i, r2]
                            - best
                        )
                    beta_cur[p_i, r_i] = np.float32(best + np.log(total))
                else:
                    beta_cur[p_i, r_i] = neg
        values = alpha[t_i - 1] + beta_cur
        best = np.max(values)
        total = 0.0
        for p_i in range(p_count):
            acc = 0.0
            for r_i in range(r_count):
                acc += np.exp(values[p_i, r_i] - best)
            post_p[t_i - 1, p_i] = acc
            total += acc
        post_p[t_i - 1] /= total
        for p_i in range(p_count):
            for r_i in range(r_count):
                beta_next[p_i, r_i] = beta_cur[p_i, r_i]
    return post_p, loglik


def run_student_t_residual_offset_hmm(
    horizontal_without_truth: pd.DataFrame,
    typewell: pd.DataFrame,
    geop_tvt: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    prepared = prepare_student_t_hmm_inputs(
        horizontal_without_truth, typewell, geop_tvt, config
    )
    hmm = get_nested(config, "model.hmm") or {}
    posterior, loglik = _hmm2_fb_student_t(
        prepared["emission_ll"],
        prepared["dm"].astype(np.float64),
        float(hmm["step"]),
        prepared["rates"],
        float(hmm["sig_r"]),
        float(hmm["sig_p"]),
        float(prepared["start_p"]),
        float(hmm["start_sig"]),
        float(prepared["r0"]),
        float(hmm["r0_sig"]),
        float(hmm["lam"]),
        float(hmm["mom"]),
    )
    grid = prepared["grid"]
    delta_mean = posterior @ grid
    variance = posterior @ (grid**2) - delta_mean**2
    std = np.sqrt(np.maximum(variance, 0.0))
    del posterior
    gc.collect()
    return {
        **prepared,
        "delta_mean": np.asarray(delta_mean, dtype=np.float64),
        "mean": np.asarray(geop_tvt, dtype=np.float64)
        + np.asarray(delta_mean, dtype=np.float64),
        "std": np.asarray(std, dtype=np.float64),
        "loglik": float(loglik),
    }


def build_stage_1_candidate_for_well(
    well: str,
    data_dir: Path,
    exp226_well: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    horizontal_path = data_dir / f"{well}__horizontal_well.csv"
    typewell_path = data_dir / f"{well}__typewell.csv"
    horizontal, typewell = load_stage_1_well(well, data_dir)
    eval_index = horizontal.index[horizontal["TVT_input"].isna()].to_numpy(np.int64)
    source = exp226_well.sort_values("row_idx", kind="mergesort")
    if not np.array_equal(eval_index, source["row_idx"].to_numpy(np.int64)):
        raise ValueError(f"exp226 row alignment failed for well={well}")
    geop = source["tvt_geop"].to_numpy(np.float64)
    started = time.time()
    result = run_student_t_residual_offset_hmm(horizontal, typewell, geop, config)
    frame = pd.DataFrame(
        {
            "id": (str(well) + "_" + source["row_idx"].astype(str)).to_numpy(),
            "well": str(well),
            "row_idx": eval_index.astype(np.int32),
            "fold": source["fold"].to_numpy(np.int8),
            "tvt_geop": geop,
            "student_t_residual_offset_hmm": result["mean"],
            "student_t_residual_offset_delta_mean": result["delta_mean"].astype(
                np.float32
            ),
            "student_t_residual_offset_hmm_std": result["std"].astype(np.float32),
            "student_t_residual_offset_hmm_loglik": np.float64(result["loglik"]),
        }
    )
    finite = np.isfinite(
        frame[
            [
                "tvt_geop",
                "student_t_residual_offset_hmm",
                "student_t_residual_offset_delta_mean",
                "student_t_residual_offset_hmm_std",
            ]
        ].to_numpy(np.float64)
    ).all()
    meta = {
        "well": str(well),
        "rows": len(frame),
        "fold": int(frame["fold"].iloc[0]),
        "status": "ok" if finite else "non_finite",
        "prefix_rows": int(result["prefix_rows"]),
        "prefix_sigma": float(result["prefix_sigma"]),
        "grid_min": float(result["grid_min"]),
        "grid_max": float(result["grid_max"]),
        "grid_size": int(len(result["grid"])),
        "delta_grid_coverage_rows": int(result["delta_grid_coverage_rows"]),
        "delta_grid_rows": int(result["delta_grid_rows"]),
        "delta_grid_coverage": float(
            result["delta_grid_coverage_rows"] / result["delta_grid_rows"]
        ),
        "delta_zero_quantization_error_ft": float(
            result["delta_zero_quantization_error_ft"]
        ),
        "native_typewell_state_coverage": float(
            result["native_typewell_state_coverage"]
        ),
        "emission_finite_coverage": float(result["emission_finite_coverage"]),
        "delta_mean_abs_median": float(np.median(np.abs(result["delta_mean"]))),
        "delta_mean_abs_max": float(np.max(np.abs(result["delta_mean"]))),
        "posterior_std_mean": float(np.mean(result["std"])),
        "posterior_std_p90": float(np.quantile(result["std"], 0.90)),
        "loglik": float(result["loglik"]),
        "elapsed_seconds": float(time.time() - started),
        "horizontal_sha256": sha256_path(horizontal_path),
        "typewell_sha256": sha256_path(typewell_path),
    }
    return frame, meta


# %% [markdown]
# ## 10. Stage 1 saved-parent evaluation and artifact guards


# %%
def load_exp281_post_freeze_control(
    config: dict[str, Any], *, candidate_content_sha256: str
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not candidate_content_sha256:
        raise ValueError("saved exp281 control may load only after candidate SHA freeze")
    spec = get_nested(config, "data.exp281_control") or {}
    path = resolve_existing(
        str(spec["filename"]), [str(value) for value in spec["candidates"]]
    )
    decompressed_sha = sha256_gzip_decompressed(path)
    if decompressed_sha != str(spec["expected_decompressed_sha256"]):
        raise ValueError("saved exp281 OOF decompressed SHA changed")
    columns = [str(value) for value in spec["safe_post_freeze_columns"]]
    frame = pd.read_csv(
        path,
        usecols=columns,
        dtype={"id": str, "well": str},
        float_precision="round_trip",
    )
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    frame["row_idx"] = pd.to_numeric(frame["row_idx"], errors="raise").astype(np.int32)
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(np.int8)
    frame["tvt_geop"] = pd.to_numeric(frame["tvt_geop"], errors="raise").astype(np.float64)
    frame["residual_offset_delta_mean"] = pd.to_numeric(
        frame["residual_offset_delta_mean"], errors="raise"
    ).astype(np.float32)
    frame["residual_offset_hmm"] = pd.to_numeric(
        frame["residual_offset_hmm"], errors="raise"
    ).astype(np.float64)
    frame["residual_offset_hmm_std"] = pd.to_numeric(
        frame["residual_offset_hmm_std"], errors="raise"
    ).astype(np.float32)
    numeric_columns = [
        "exp226_pred",
        "exp263_fixed",
        "md_since",
        "true_tvt_readout_only",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.float64)
    frame = frame.sort_values(["well", "row_idx"], kind="mergesort").reset_index(drop=True)
    if frame["id"].duplicated().any():
        raise ValueError("saved exp281 control has duplicate ids")
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(frame) != expected_rows or frame["well"].nunique() != expected_wells:
        raise ValueError("saved exp281 control coverage changed")
    parent_content_columns = [
        "id",
        "row_idx",
        "fold",
        "tvt_geop",
        "residual_offset_delta_mean",
        "residual_offset_hmm",
        "residual_offset_hmm_std",
    ]
    parent_content_sha = dataframe_content_sha(frame, parent_content_columns)
    if parent_content_sha != str(spec["expected_content_sha256"]):
        raise ValueError(
            "saved exp281 logical prediction content SHA changed: "
            f"{parent_content_sha} != {spec['expected_content_sha256']}"
        )
    frame = frame.rename(
        columns={
            "residual_offset_delta_mean": "gaussian_residual_offset_delta_mean",
            "residual_offset_hmm": "gaussian_residual_offset_hmm",
            "residual_offset_hmm_std": "gaussian_residual_offset_hmm_std",
        }
    )
    manifest = {
        "name": "exp281_saved_gaussian_residual_offset_hmm_post_candidate_freeze",
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": sha256_path(path),
        "decompressed_sha256": decompressed_sha,
        "content_sha256": parent_content_sha,
        "rows": len(frame),
        "wells": int(frame["well"].nunique()),
        "candidate_content_sha256_before_load": candidate_content_sha256,
    }
    return frame, manifest


def merge_stage_1_control_after_candidate_freeze(
    candidate: pd.DataFrame,
    control: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    candidate = candidate.sort_values(["well", "row_idx"], kind="mergesort").reset_index(
        drop=True
    )
    control = control.sort_values(["well", "row_idx"], kind="mergesort").reset_index(
        drop=True
    )
    if len(candidate) != len(control):
        raise ValueError("Stage 1 candidate/control row count mismatch")
    identity_match = (
        np.array_equal(candidate["id"].to_numpy(), control["id"].to_numpy())
        and np.array_equal(candidate["well"].to_numpy(), control["well"].to_numpy())
        and np.array_equal(candidate["row_idx"].to_numpy(), control["row_idx"].to_numpy())
        and np.array_equal(candidate["fold"].to_numpy(), control["fold"].to_numpy())
        and np.array_equal(
            candidate["tvt_geop"].to_numpy(np.float64),
            control["tvt_geop"].to_numpy(np.float64),
        )
    )
    if not identity_match:
        raise ValueError("Stage 1 candidate and saved exp281 identities differ")
    row_identity_coverage = 1.0
    keep = [
        "id",
        "well",
        "row_idx",
        "gaussian_residual_offset_delta_mean",
        "gaussian_residual_offset_hmm",
        "gaussian_residual_offset_hmm_std",
        "exp226_pred",
        "exp263_fixed",
        "md_since",
        "true_tvt_readout_only",
    ]
    merged = candidate.merge(
        control[keep],
        on=["id", "well", "row_idx"],
        how="left",
        validate="one_to_one",
    )
    required = [
        "student_t_residual_offset_hmm",
        "gaussian_residual_offset_hmm",
        "exp226_pred",
        "exp263_fixed",
        "md_since",
        "true_tvt_readout_only",
    ]
    if not np.isfinite(merged[required].to_numpy(np.float64)).all():
        raise ValueError("Stage 1 post-freeze control join produced non-finite values")
    return merged, row_identity_coverage


def stage_1_score_prediction(
    truth: np.ndarray | pd.Series, prediction: np.ndarray | pd.Series
) -> dict[str, Any]:
    truth_array = np.asarray(truth, dtype=np.float64)
    prediction_array = np.asarray(prediction, dtype=np.float64)
    error = prediction_array - truth_array
    if len(error) == 0 or not np.isfinite(error).all():
        raise ValueError("Stage 1 metric requires non-empty finite errors")
    return {
        "rows": len(error),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "within5": float(np.mean(np.abs(error) <= 5.0)),
        "within10": float(np.mean(np.abs(error) <= 10.0)),
    }


def build_stage_1_candidate_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    truth = frame["true_tvt_readout_only"]
    return pd.DataFrame(
        [
            {
                "candidate": candidate,
                **stage_1_score_prediction(truth, frame[candidate]),
            }
            for candidate in STAGE_1_CANDIDATES
        ]
    )


def build_stage_1_fold_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold, part in frame.groupby("fold", sort=True):
        truth = part["true_tvt_readout_only"]
        for candidate in STAGE_1_CANDIDATES:
            rows.append(
                {
                    "fold": int(fold),
                    "candidate": candidate,
                    **stage_1_score_prediction(truth, part[candidate]),
                }
            )
    return pd.DataFrame(rows)


def build_stage_1_scope_metrics(
    frame: pd.DataFrame,
    hidden_assignments: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    scopes: list[tuple[str, pd.DataFrame]] = [
        ("overall", frame),
        ("long_tail_1000_plus", frame.loc[frame["md_since"] >= 1000.0]),
    ]
    role_columns = get_nested(config, "data.hidden_like.role_columns") or {}
    if hidden_assignments.empty:
        raise ValueError("Stage 1 requires hidden-like assignments")
    role_by_well = hidden_assignments.set_index("well_id")
    for scope_name, role_column in role_columns.items():
        valid_wells = set(
            role_by_well.index[
                role_by_well[str(role_column)].astype(str) == "valid"
            ].astype(str)
        )
        scopes.append(
            (
                str(scope_name),
                frame.loc[frame["well"].astype(str).isin(valid_wells)],
            )
        )
    rows: list[dict[str, Any]] = []
    for scope, part in scopes:
        if part.empty:
            raise ValueError(f"Stage 1 scope {scope} selected zero rows")
        truth = part["true_tvt_readout_only"]
        parent = stage_1_score_prediction(
            truth, part["gaussian_residual_offset_hmm"]
        )
        student = stage_1_score_prediction(
            truth, part["student_t_residual_offset_hmm"]
        )
        rows.append(
            {
                "scope": scope,
                "rows": len(part),
                "wells": int(part["well"].nunique()),
                "gaussian_parent_rmse": parent["rmse"],
                "student_t_rmse": student["rmse"],
                "student_t_minus_gaussian_rmse": student["rmse"] - parent["rmse"],
                "student_t_gain_ft": parent["rmse"] - student["rmse"],
                "gaussian_parent_mae": parent["mae"],
                "student_t_mae": student["mae"],
            }
        )
    return pd.DataFrame(rows)


def build_stage_1_by_well_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well, part in frame.groupby("well", sort=True):
        truth = part["true_tvt_readout_only"]
        parent = stage_1_score_prediction(
            truth, part["gaussian_residual_offset_hmm"]
        )
        student = stage_1_score_prediction(
            truth, part["student_t_residual_offset_hmm"]
        )
        rows.append(
            {
                "well": str(well),
                "fold": int(part["fold"].iloc[0]),
                "rows": len(part),
                "gaussian_parent_rmse": parent["rmse"],
                "student_t_rmse": student["rmse"],
                "delta_rmse_vs_exp281": student["rmse"] - parent["rmse"],
                "gaussian_parent_mae": parent["mae"],
                "student_t_mae": student["mae"],
            }
        )
    return pd.DataFrame(rows)


def evaluate_stage_1_guard(
    candidate_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    scope_metrics: pd.DataFrame,
    by_well: pd.DataFrame,
    *,
    finite_coverage: float,
    row_identity_coverage: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    guards = get_nested(config, "validation.stage_1_pass_requires_all") or {}
    metric_by_candidate = candidate_metrics.set_index("candidate")
    parent_rmse = float(
        metric_by_candidate.loc["gaussian_residual_offset_hmm", "rmse"]
    )
    candidate_rmse = float(
        metric_by_candidate.loc["student_t_residual_offset_hmm", "rmse"]
    )
    exp226_rmse = float(metric_by_candidate.loc["exp226_pred", "rmse"])
    expected_parent = float(
        get_nested(config, "data.exp281_control.expected_parent_rmse")
    )
    expected_exp226 = float(
        get_nested(config, "data.exp281_control.expected_exp226_rmse")
    )
    parity_atol = float(guards["required_parent_rmse_parity_atol_ft"])
    folds = fold_metrics.pivot(index="fold", columns="candidate", values="rmse")
    improved_folds = int(
        (
            folds["student_t_residual_offset_hmm"]
            < folds["gaussian_residual_offset_hmm"]
        ).sum()
    )
    scope_by_name = scope_metrics.set_index("scope")
    required_scopes = [str(value) for value in guards["required_scopes"]]
    scope_deltas = {
        scope: float(scope_by_name.loc[scope, "student_t_minus_gaussian_rmse"])
        for scope in required_scopes
    }
    by_well_delta = by_well["delta_rmse_vs_exp281"].to_numpy(np.float64)
    by_well_p95 = float(np.quantile(by_well_delta, 0.95))
    worst_well_index = int(np.argmax(by_well_delta))
    worst_well = str(by_well.iloc[worst_well_index]["well"])
    worst_delta = float(by_well_delta[worst_well_index])
    checks = {
        "exp281_parent_rmse_parity": bool(
            abs(parent_rmse - expected_parent) <= parity_atol
        ),
        "exp226_rmse_parity": bool(abs(exp226_rmse - expected_exp226) <= parity_atol),
        "overall_gain_vs_exp281": bool(
            parent_rmse - candidate_rmse
            >= float(guards["minimum_rmse_gain_vs_exp281_ft"])
        ),
        "improved_folds": bool(
            improved_folds >= int(guards["minimum_improved_folds"])
        ),
        "required_scopes_non_regression": bool(
            all(
                delta <= float(guards["maximum_scope_rmse_regression_ft"])
                for delta in scope_deltas.values()
            )
        ),
        "by_well_p95_non_regression": bool(
            by_well_p95 <= float(guards["maximum_by_well_p95_delta_rmse_ft"])
        ),
        "worst_well_regression": bool(
            worst_delta <= float(guards["maximum_worst_well_regression_ft"])
        ),
        "finite_coverage": bool(
            finite_coverage >= float(guards["required_finite_coverage"])
        ),
        "row_identity_coverage": bool(
            row_identity_coverage
            >= float(guards["required_row_identity_coverage"])
        ),
    }
    stage_1_passed = bool(all(checks.values()))
    direct_rmse_passed = bool(
        candidate_rmse <= float(guards["maximum_direct_rmse_for_promotion"])
    )
    return {
        "passed": stage_1_passed,
        "direct_promotion_passed": bool(stage_1_passed and direct_rmse_passed),
        "checks": checks,
        "direct_promotion_check": {
            "passed": direct_rmse_passed,
            "candidate_rmse": candidate_rmse,
            "maximum_rmse": float(guards["maximum_direct_rmse_for_promotion"]),
        },
        "candidate_rmse": candidate_rmse,
        "parent_rmse": parent_rmse,
        "exp226_rmse": exp226_rmse,
        "gain_vs_exp281_ft": parent_rmse - candidate_rmse,
        "delta_vs_exp226_ft": candidate_rmse - exp226_rmse,
        "improved_folds": improved_folds,
        "scope_delta_rmse_vs_exp281": scope_deltas,
        "by_well_p95_delta_rmse_vs_exp281": by_well_p95,
        "worst_well": worst_well,
        "worst_well_delta_rmse_vs_exp281": worst_delta,
        "finite_coverage": finite_coverage,
        "row_identity_coverage": row_identity_coverage,
        "decision": (
            "stage_1_passed_direct_promotion"
            if stage_1_passed and direct_rmse_passed
            else (
                "stage_1_scientific_passed_direct_promotion_failed"
                if stage_1_passed
                else "stage_1_failed_close_without_rescue"
            )
        ),
        "stage_0_prerequisite_passed": False,
        "execution_basis": "explicit_user_override_after_stage_0_fail",
    }


def run_stage_1_experiment(config: dict[str, Any]) -> dict[str, Any]:
    if not KAGGLE_WORKING_ROOT.exists() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError(
            "Full exp342 Stage 1 must run on Kaggle. EXPERIMENT_ALLOW_LOCAL=1 is "
            "reserved for an explicitly approved local smoke run."
        )
    validate_scientific_contract(config, require_run_approval=True)
    if not bool(get_nested(config, "execution.run_stage_1")):
        raise RuntimeError("exp342 Stage 1 execution is not enabled")
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba is required for exp342 Stage 1 exact HMM")
    set_num_threads(int(get_nested(config, "runtime.numba_num_threads") or 1))
    started = time.time()
    safe_oof, _, exp226_manifest = load_exp226_safe(config)
    data_dir = train_data_dir(config)
    wells = list_stage_1_wells(data_dir)
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(wells) != expected_wells or set(wells) != set(safe_oof["well_id"].unique()):
        raise ValueError("raw-well and exp226 well sets do not match for Stage 1")
    grouped_exp226 = {
        str(well): part for well, part in safe_oof.groupby("well_id", sort=False)
    }
    frames: list[pd.DataFrame] = []
    well_rows: list[dict[str, Any]] = []
    progress_every = 2
    for index, well in enumerate(wells, start=1):
        print(f"[exp342 Stage 1] {index}/{len(wells)} well={well}", flush=True)
        frame, meta = build_stage_1_candidate_for_well(
            well, data_dir, grouped_exp226[well], config
        )
        frames.append(frame)
        well_rows.append(meta)
        if index == 1 or index % progress_every == 0 or index == len(wells):
            print(json.dumps(to_jsonable(meta), sort_keys=True), flush=True)
        gc.collect()
    candidate = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["well", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    del frames, grouped_exp226, safe_oof
    gc.collect()
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    if len(candidate) != expected_rows or candidate["well"].nunique() != expected_wells:
        raise ValueError("Stage 1 full candidate coverage mismatch")
    if candidate["id"].duplicated().any():
        raise ValueError("Stage 1 candidate contains duplicate ids")
    candidate_content_columns = [
        "id",
        "row_idx",
        "fold",
        "tvt_geop",
        "student_t_residual_offset_delta_mean",
        "student_t_residual_offset_hmm",
        "student_t_residual_offset_hmm_std",
    ]
    candidate_content_sha = dataframe_content_sha(candidate, candidate_content_columns)
    if not candidate_content_sha:
        raise RuntimeError("failed to freeze Stage 1 candidate content SHA")

    # Parent paths and truth are first read after every Student-t path is frozen.
    control, exp281_manifest = load_exp281_post_freeze_control(
        config, candidate_content_sha256=candidate_content_sha
    )
    merged, row_identity_coverage = merge_stage_1_control_after_candidate_freeze(
        candidate, control
    )
    del candidate, control
    gc.collect()

    hidden_assignments, hidden_manifest = load_hidden_like_assignments(config)
    candidate_metrics = build_stage_1_candidate_metrics(merged)
    fold_metrics = build_stage_1_fold_metrics(merged)
    scope_metrics = build_stage_1_scope_metrics(
        merged, hidden_assignments, config
    )
    by_well = build_stage_1_by_well_metrics(merged)
    well_manifest = pd.DataFrame(well_rows).sort_values(
        "well", kind="mergesort"
    ).reset_index(drop=True)
    finite_columns = [
        "student_t_residual_offset_hmm",
        "student_t_residual_offset_delta_mean",
        "student_t_residual_offset_hmm_std",
    ]
    finite_coverage = float(
        np.isfinite(merged[finite_columns].to_numpy(np.float64)).mean()
    )
    guard = evaluate_stage_1_guard(
        candidate_metrics,
        fold_metrics,
        scope_metrics,
        by_well,
        finite_coverage=finite_coverage,
        row_identity_coverage=row_identity_coverage,
        config=config,
    )

    artifacts = artifact_dir()
    paths = {
        "predictions": artifacts / f"{OUTPUT_PREFIX}_stage1_oof_predictions.csv.gz",
        "candidate_metrics": artifacts / f"{OUTPUT_PREFIX}_stage1_candidate_metrics.csv",
        "fold_metrics": artifacts / f"{OUTPUT_PREFIX}_stage1_fold_metrics.csv",
        "scope_metrics": artifacts / f"{OUTPUT_PREFIX}_stage1_scope_metrics.csv",
        "by_well_metrics": artifacts / f"{OUTPUT_PREFIX}_stage1_by_well_metrics.csv",
        "well_manifest": artifacts / f"{OUTPUT_PREFIX}_stage1_well_manifest.csv",
        "input_manifest": artifacts / f"{OUTPUT_PREFIX}_stage1_input_manifest.csv",
        "decoder_manifest": artifacts / f"{OUTPUT_PREFIX}_stage1_decoder_manifest.json",
        "gate": artifacts / f"{OUTPUT_PREFIX}_stage1_gate.json",
        "summary": artifacts / f"{OUTPUT_PREFIX}_stage1_summary.json",
    }
    merged.to_csv(
        paths["predictions"],
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    candidate_metrics.to_csv(paths["candidate_metrics"], index=False)
    fold_metrics.to_csv(paths["fold_metrics"], index=False)
    scope_metrics.to_csv(paths["scope_metrics"], index=False)
    by_well.to_csv(paths["by_well_metrics"], index=False)
    well_manifest.to_csv(paths["well_manifest"], index=False)
    input_manifest = pd.DataFrame(
        [
            exp226_manifest,
            exp281_manifest,
            hidden_manifest,
            {
                "name": "raw_train_well_files",
                "path": str(data_dir),
                "rows": int(well_manifest["rows"].sum()),
                "wells": len(well_manifest),
                "raw_sha256": dataframe_content_sha(
                    well_manifest,
                    ["well", "horizontal_sha256", "typewell_sha256"],
                ),
            },
        ]
    )
    input_manifest.to_csv(paths["input_manifest"], index=False)
    decoder_manifest = {
        "experiment": EXPERIMENT_NAME,
        "parent": get_nested(config, "lineage.parent"),
        "stage": "stage_1_explicit_override",
        "stage_0_prerequisite_passed": False,
        "execution_basis": "explicit_user_override_after_stage_0_fail",
        "hmm": get_nested(config, "model.hmm"),
        "coordinate_contract": {
            "equation": "TVT_t = exp226_tvt_geop_t + delta_t",
            "absolute_transition_center": "diff(exp226_tvt_geop)",
            "delta_transition_center": "offset_rate_t * delta_md_t",
            "truth_available_to_decoder": False,
            "exp226_tvt_pred_available_to_decoder": False,
        },
        "emission_change_only": {
            "parent": "gaussian_minus_half_min_z2_600",
            "candidate": "student_t_df4_minus_2p5_log1p_z2_over_4",
            "additional_clip": "none",
        },
        "truth_attachment": "after_all_student_t_well_paths_and_content_sha_freeze",
        "parent_control": "sha_pinned_saved_exp281_oof_no_hmm_rerun",
        "candidate_count": 1,
    }
    write_json(paths["decoder_manifest"], decoder_manifest)
    write_json(paths["gate"], guard)
    prediction_decompressed_sha = sha256_gzip_decompressed(paths["predictions"])
    file_sha = {
        key: sha256_path(path)
        for key, path in paths.items()
        if key not in {"summary"}
    }
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": (
            "stage_1_completed_guard_passed"
            if guard["passed"]
            else "stage_1_completed_guard_failed"
        ),
        "decision": guard["decision"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "route": get_nested(config, "experiment.route"),
        "stage_0_prerequisite_passed": False,
        "execution_basis": "explicit_user_override_after_stage_0_fail",
        "rows": len(merged),
        "wells": int(merged["well"].nunique()),
        "active_hmm_variants": 1,
        "hmm_well_runs": len(wells),
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "parent_control_retraining": False,
        "gpu": False,
        "inference": False,
        "submission": False,
        "elapsed_seconds": float(time.time() - started),
        "candidate_metrics": candidate_metrics.to_dict("records"),
        "fold_metrics": fold_metrics.to_dict("records"),
        "scope_metrics": scope_metrics.to_dict("records"),
        "promotion_guard": guard,
        "candidate_content_sha256": candidate_content_sha,
        "decoder_manifest_sha256": mapping_sha256(decoder_manifest),
        "artifacts": {key: str(path) for key, path in paths.items()},
        "sha256": {
            "predictions_raw_gzip": sha256_path(paths["predictions"]),
            "predictions_decompressed": prediction_decompressed_sha,
            **file_sha,
        },
    }
    write_json(paths["summary"], summary)
    write_json(
        metrics_output_path(),
        {
            "experiment": EXPERIMENT_NAME,
            "status": summary["status"],
            "metric": "rmse_tvt",
            "cv": guard["candidate_rmse"],
            "public_lb": None,
            "private_lb": None,
            "stage_0_prerequisite_passed": False,
            "execution_basis": "explicit_user_override_after_stage_0_fail",
            "promotion_guard": guard,
            "candidate_metrics": candidate_metrics.to_dict("records"),
            "candidate_content_sha256": candidate_content_sha,
            "decoder_manifest_sha256": summary["decoder_manifest_sha256"],
            "inference": False,
            "submission": False,
        },
    )
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    return summary


# %% [markdown]
# ## 11. Setup and contract preview


# %%
CONFIG: dict[str, Any] | None = None
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    validate_scientific_contract(CONFIG, require_run_approval=True)
    print(
        json.dumps(
            {
                "experiment": get_nested(CONFIG, "experiment.name"),
                "route": get_nested(CONFIG, "experiment.route"),
                "parent": get_nested(CONFIG, "lineage.parent"),
                "stage": get_nested(CONFIG, "execution.active_stage"),
                "student_t_df": get_nested(CONFIG, "audit.student_t_emission.degrees_of_freedom"),
                "gaussian_control": get_nested(CONFIG, "model.stage_0.gaussian_control"),
                "shift_bank_ft": get_nested(CONFIG, "audit.shift_bank_ft"),
                "block_rows": get_nested(CONFIG, "audit.block_rows"),
                "stage_0_contract": get_nested(CONFIG, "execution_contract.stage_0"),
                "stage_1_contract": get_nested(
                    CONFIG, "execution_contract.stage_1_override"
                ),
                "stage_1_implemented": get_nested(CONFIG, "implementation.stage_1_implemented"),
                "stage_1_override_approved": get_nested(
                    CONFIG, "execution.stage_1_override_approved"
                ),
                "run_stage_0": get_nested(CONFIG, "execution.run_stage_0"),
                "run_stage_1": get_nested(CONFIG, "execution.run_stage_1"),
                "hmm": get_nested(CONFIG, "model.hmm"),
                "inference": get_nested(CONFIG, "execution.run_inference"),
                "submission": get_nested(CONFIG, "execution.create_submission"),
                "numba_available": NUMBA_AVAILABLE,
            },
            indent=2,
        )
    )


# %% [markdown]
# ## 12. Run the explicitly selected stage


# %%
if EXECUTE_NOTEBOOK:
    assert CONFIG is not None
    if bool(get_nested(CONFIG, "execution.run_stage_1")):
        EXP342_SUMMARY = run_stage_1_experiment(CONFIG)
    elif bool(get_nested(CONFIG, "execution.run_stage_0")):
        EXP342_SUMMARY = run_stage_0_experiment(CONFIG)
    else:
        raise RuntimeError("exp342 has no approved execution stage")
