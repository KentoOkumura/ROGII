# %% [markdown]
# # exp365 bounded GR registration-offset HMM — Stage 0 train-side preflight
#
# This compact self-contained notebook implements only the design-frozen
# known-prefix Stage 0. It separates physical TVT from a five-state GR lookup
# offset, evaluates sequential held-out GR NLL without reading suffix truth,
# freezes the rolling ledger and offset posterior by content SHA, and applies
# the fixed scientific and resource gates. It does not decode the expanded
# exact HMM, rerun exp209, train a model, or create a submission.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Notebook-safe runtime, configuration, path, and SHA helpers
# 3. Frozen scientific and execution contract
# 4. Visible-prefix input and registration-filter helpers
# 5. Rolling-origin Stage 0 generation
# 6. Five-state exact-HMM resource projection and SHA freeze
# 7. Fold metrics and promotion gates
# 8. Execution orchestration and generated artifacts
# 9. Setup and configuration preview
# 10. Fail-closed Stage 0 execution selection

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold

try:
    from IPython import get_ipython
    from IPython.display import display
except ImportError:  # pragma: no cover
    def get_ipython() -> None:
        return None

    def display(value: Any) -> None:
        print(value)


EXPERIMENT_NAME = "exp365_bounded_gr_registration_offset_hmm"
OUTPUT_PREFIX = f"{EXPERIMENT_NAME}_stage0"
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
IMPORT_ONLY = os.environ.get("EXP365_IMPORT_ONLY", "0") == "1"
EXECUTE_NOTEBOOK = get_ipython() is not None and not IMPORT_ONLY
FORBIDDEN_HORIZONTAL_COLUMNS = {
    "TVT",
    "target",
    "true_tvt",
    "error",
    "abs_error",
}


# %% [markdown]
# ## 2. Notebook-safe runtime, configuration, path, and SHA helpers

# %%
def get_nested(mapping: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported JSON value: {type(value)!r}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=json_default,
    )


def mapping_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_decompressed_gzip(path: str | Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(Path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_sha_report(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    report: dict[str, Any] = {
        "path": str(resolved),
        "bytes": int(resolved.stat().st_size),
        "raw_sha256": sha256_file(resolved),
    }
    if resolved.suffix == ".gz":
        report["decompressed_sha256"] = sha256_decompressed_gzip(resolved)
        report["content_sha256"] = report["decompressed_sha256"]
    else:
        report["content_sha256"] = report["raw_sha256"]
    return report


def dataframe_schema_sha256(frame: pd.DataFrame) -> str:
    return mapping_sha256(
        [
            {"name": str(column), "dtype": str(frame[column].dtype)}
            for column in frame.columns
        ]
    )


def dataframe_content_sha256(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> str:
    selected = frame if columns is None else frame.loc[:, list(columns)]
    text = selected.to_csv(index=False, float_format="%.10g", lineterminator="\n")
    return hashlib.sha256(text.encode()).hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(value),
            indent=2,
            ensure_ascii=False,
            default=json_default,
        )
        + "\n"
    )


def write_deterministic_csv_gzip(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        float_format="%.10g",
        lineterminator="\n",
    )


def project_root() -> Path:
    for candidate in (PACKAGE_DIR, *PACKAGE_DIR.parents):
        if (candidate / "project.yml").is_file():
            return candidate
    return PACKAGE_DIR


def candidate_package_dirs() -> list[Path]:
    root = project_root()
    candidates = [
        PACKAGE_DIR,
        root / "experiments" / EXPERIMENT_NAME,
        KAGGLE_WORKING_ROOT,
    ]
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(
            path.parent
            for path in sorted(KAGGLE_INPUT_ROOT.glob("**/config.yaml"))
            if path.parent.name == EXPERIMENT_NAME
        )
    return candidates


def resolve_package_dir() -> Path:
    for candidate in candidate_package_dirs():
        path = candidate / "config.yaml"
        if not path.is_file():
            continue
        try:
            value = yaml.safe_load(path.read_text()) or {}
        except (OSError, yaml.YAMLError):
            continue
        if get_nested(value, "experiment.name") == EXPERIMENT_NAME:
            return candidate
    raise FileNotFoundError(f"Could not locate config.yaml for {EXPERIMENT_NAME}")


def load_config(package_dir: Path | None = None) -> dict[str, Any]:
    directory = resolve_package_dir() if package_dir is None else Path(package_dir)
    value = yaml.safe_load((directory / "config.yaml").read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError("config.yaml must contain a mapping")
    return value


def is_kaggle_runtime() -> bool:
    return KAGGLE_INPUT_ROOT.exists() and KAGGLE_WORKING_ROOT.exists()


def require_authoritative_runtime() -> None:
    if is_kaggle_runtime() or os.environ.get("EXPERIMENT_ALLOW_LOCAL", "0") == "1":
        return
    raise RuntimeError(
        "exp365 Stage 0 must run on Kaggle. EXPERIMENT_ALLOW_LOCAL=1 is reserved "
        "for an explicitly approved local smoke run."
    )


def resolve_train_dir(config: Mapping[str, Any]) -> Path:
    configured = Path(str(get_nested(config, "data.train_dir", "data/raw/train")))
    candidates = [
        configured,
        Path.cwd() / configured,
        project_root() / configured,
    ]
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(
            [
                KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction" / "train",
                *sorted(KAGGLE_INPUT_ROOT.glob("**/train")),
            ]
        )
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*__horizontal_well.csv")):
            return candidate
    raise FileNotFoundError("Could not resolve raw train directory")


def output_dirs(package_dir: Path) -> tuple[Path, Path]:
    root = KAGGLE_WORKING_ROOT if is_kaggle_runtime() else package_dir
    artifacts = root / "artifacts"
    features = root / "features"
    artifacts.mkdir(parents=True, exist_ok=True)
    features.mkdir(parents=True, exist_ok=True)
    return artifacts, features


# %% [markdown]
# ## 3. Frozen scientific and execution contract

# %%
def stage0_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    stage0 = get_nested(config, "validation.stage_0", {})
    offset = get_nested(config, "model.gr_registration_offset", {})
    fixed = get_nested(config, "model.fixed_from_exp209", {})
    execution = get_nested(config, "execution", {})
    contract = {
        "experiment": get_nested(config, "experiment.name"),
        "route": get_nested(config, "experiment.route"),
        "implementation_scope": execution.get("implementation_scope"),
        "run_stage_1": execution.get("run_stage_1"),
        "history_rows": stage0.get("history_rows"),
        "heldout_rows": stage0.get("heldout_rows"),
        "stride_rows": stage0.get("stride_rows"),
        "minimum_known_prefix_rows": stage0.get("minimum_known_prefix_rows"),
        "posterior_snapshot": stage0.get("posterior_snapshot"),
        "predictive_policy": stage0.get("predictive_policy"),
        "sigma_policy": stage0.get("sigma_policy"),
        "nll": stage0.get("nll"),
        "fold_assignment": stage0.get("fold_assignment"),
        "fold_pass_definition": stage0.get("fold_pass_definition"),
        "negative_control": stage0.get("negative_control"),
        "resource_projection_wells": stage0.get("resource_projection_wells"),
        "resource_projection": stage0.get("resource_projection"),
        "offset_values_ft": offset.get("values_ft"),
        "offset_initial_probability": offset.get("initial_probability"),
        "offset_transition_matrix": offset.get("transition_matrix"),
        "physical_output": offset.get("physical_output"),
        "emission_lookup_position": offset.get("emission_lookup_position"),
        "position_grid_step_ft": fixed.get("position_grid_step_ft"),
        "n_rates": fixed.get("n_rates"),
        "emission": fixed.get("emission"),
        "emission_squared_z_clip": fixed.get("emission_squared_z_clip"),
        "gr_sigma_min": fixed.get("gr_sigma_min"),
        "gr_sigma_max": fixed.get("gr_sigma_max"),
        "gr_sigma_default": fixed.get("gr_sigma_default"),
        "band_pad_ft": fixed.get("band_pad_ft"),
        "typewell_outer_pad_ft": fixed.get("typewell_outer_pad_ft"),
        "output": fixed.get("output"),
        "hmm_well_runs": stage0.get("hmm_well_runs"),
    }
    contract["contract_sha256"] = mapping_sha256(contract)
    return contract


def expected_offset_transition() -> np.ndarray:
    move = 1.0 / 512.0
    transition = np.zeros((5, 5), dtype=np.float64)
    for state in range(5):
        transition[state, state] = 1.0 - 2.0 * move
        for direction in (-1, 1):
            target = state + direction
            if 0 <= target < 5:
                transition[state, target] += move
            else:
                transition[state, state] += move
    return transition


def validate_scientific_contract(
    config: Mapping[str, Any],
    *,
    require_run_approval: bool = False,
) -> dict[str, Any]:
    contract = stage0_contract(config)
    required_equal = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "implementation_scope": "stage0_only",
        "run_stage_1": False,
        "history_rows": 128,
        "heldout_rows": 64,
        "stride_rows": 64,
        "minimum_known_prefix_rows": 256,
        "posterior_snapshot": "end_of_history_before_heldout",
        "predictive_policy": (
            "sequential_one_step_ahead_update_on_observed_gr_only"
        ),
        "sigma_policy": "exp209_std_on_history_only_interpolated_gr_clip_10_60",
        "nll": "normalized_gaussian_with_squared_z_clip_600",
        "fold_assignment": "sklearn_group_kfold_on_stable_well_order",
        "fold_pass_definition": (
            "real_nll_gain_at_least_1pct_and_real_minus_circular_gain_at_least_0p5pct"
        ),
        "resource_projection_wells": 16,
        "offset_values_ft": [-6.0, -3.0, 0.0, 3.0, 6.0],
        "offset_initial_probability": [0.05, 0.15, 0.60, 0.15, 0.05],
        "physical_output": "physical_position",
        "emission_lookup_position": (
            "physical_position_plus_gr_registration_offset"
        ),
        "position_grid_step_ft": 0.35,
        "n_rates": 41,
        "emission": "gauss",
        "emission_squared_z_clip": 600.0,
        "gr_sigma_min": 10.0,
        "gr_sigma_max": 60.0,
        "gr_sigma_default": 30.0,
        "band_pad_ft": 100.0,
        "typewell_outer_pad_ft": 40.0,
        "output": "posterior_mean_physical_position",
        "hmm_well_runs": 0,
    }
    mismatches = {
        key: {"expected": expected, "actual": contract.get(key)}
        for key, expected in required_equal.items()
        if contract.get(key) != expected
    }
    actual_transition = np.asarray(
        contract["offset_transition_matrix"], dtype=np.float64
    )
    if actual_transition.shape != (5, 5) or not np.array_equal(
        actual_transition, expected_offset_transition()
    ):
        mismatches["offset_transition_matrix"] = {
            "expected": expected_offset_transition().tolist(),
            "actual": actual_transition.tolist(),
        }
    if mismatches:
        raise ValueError(f"Frozen exp365 Stage 0 contract mismatch: {mismatches}")

    negative = contract["negative_control"]
    expected_negative = {
        "kind": "within_well_circular_shift_of_observed_known_prefix_gr",
        "shift_observed_values": 64,
        "preserve_missing_mask": True,
        "preserve_observed_value_multiset": True,
        "minimum_shifted_values": 2,
    }
    if negative != expected_negative:
        raise ValueError("exp365 circular negative-control contract changed")

    forbidden = set(get_nested(config, "model.forbidden", []))
    expected_forbidden = {
        "output_position_plus_offset",
        "offset_or_transition_grid",
        "dtw_or_affine_gr_transform",
        "rate_prediction_from_prefix_or_geometry",
        "emission_sigma_or_dynamics_change",
        "blend_or_selector",
        "parent_control_rerun",
    }
    if forbidden != expected_forbidden:
        raise ValueError("exp365 forbidden-operation contract changed")

    expected_counts = {
        "diagnostic_variants": 1,
        "offset_states": 5,
        "reporting_folds": 5,
        "resource_projection_wells": 16,
        "exact_hmm_well_runs": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "parent_control_reruns": 0,
    }
    if get_nested(config, "execution.stage_0_counts") != expected_counts:
        raise ValueError("exp365 Stage 0 execution-count contract changed")

    if require_run_approval and not (
        bool(get_nested(config, "execution.kaggle_push_approved"))
        and bool(get_nested(config, "execution.kaggle_execution_approved"))
        and bool(get_nested(config, "execution.run_stage_0"))
    ):
        raise PermissionError("exp365 Kaggle Stage 0 run is not approved")
    return contract


# %% [markdown]
# ## 4. Visible-prefix input and registration-filter helpers

# %%
def discover_wells(train_dir: Path, maximum_wells: int | None = None) -> list[str]:
    horizontal = {
        path.name.removesuffix("__horizontal_well.csv")
        for path in train_dir.glob("*__horizontal_well.csv")
    }
    typewell = {
        path.name.removesuffix("__typewell.csv")
        for path in train_dir.glob("*__typewell.csv")
    }
    wells = sorted(horizontal.intersection(typewell))
    if maximum_wells is not None:
        wells = wells[: int(maximum_wells)]
    return wells


def load_visible_prefix_well(
    well_id: str,
    train_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizontal_columns = ["MD", "Z", "GR", "TVT_input"]
    if FORBIDDEN_HORIZONTAL_COLUMNS.intersection(horizontal_columns):
        raise RuntimeError("Forbidden suffix-truth column requested before freeze")
    horizontal = pd.read_csv(
        train_dir / f"{well_id}__horizontal_well.csv",
        usecols=horizontal_columns,
    )
    typewell = pd.read_csv(
        train_dir / f"{well_id}__typewell.csv",
        usecols=["TVT", "GR"],
    )
    if horizontal.empty or typewell.empty:
        raise ValueError(f"Empty raw input for well={well_id}")
    return horizontal, typewell


def prepare_typewell(typewell: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    clean = typewell.copy()
    clean["TVT"] = pd.to_numeric(clean["TVT"], errors="coerce")
    clean["GR"] = pd.to_numeric(clean["GR"], errors="coerce")
    clean = (
        clean.dropna(subset=["TVT"])
        .sort_values("TVT", kind="mergesort")
        .drop_duplicates("TVT", keep="last")
        .reset_index(drop=True)
    )
    clean["GR"] = clean["GR"].ffill().bfill()
    clean = clean.dropna(subset=["GR"])
    if len(clean) < 3:
        raise ValueError("Typewell must contain at least three finite TVT/GR rows")
    return (
        clean["TVT"].to_numpy(np.float64),
        clean["GR"].to_numpy(np.float64),
    )


def fixed_rolling_windows(
    known_rows: int,
    history_rows: int,
    heldout_rows: int,
    stride_rows: int,
    minimum_known_prefix_rows: int,
) -> list[tuple[int, int, int]]:
    if known_rows < minimum_known_prefix_rows:
        return []
    stop_offset = history_rows + heldout_rows
    return [
        (start, start + history_rows, start + stop_offset)
        for start in range(0, known_rows - stop_offset + 1, stride_rows)
    ]


def circular_shift_observed_values(
    values: np.ndarray,
    shift_observed_values: int,
) -> tuple[np.ndarray, int]:
    original = np.asarray(values, dtype=np.float64)
    shifted = original.copy()
    observed_mask = np.isfinite(original)
    observed = original[observed_mask]
    if len(observed) < 2:
        return shifted, 0
    shift = int(shift_observed_values) % len(observed)
    if shift == 0:
        shift = 1
    shifted[observed_mask] = np.roll(observed, shift)
    if not np.array_equal(
        np.sort(shifted[observed_mask]),
        np.sort(original[observed_mask]),
    ):
        raise RuntimeError("Circular control did not preserve observed GR values")
    if not np.array_equal(np.isfinite(shifted), observed_mask):
        raise RuntimeError("Circular control changed the missing-GR mask")
    return shifted, shift


def fill_history_gr(
    observed_gr: np.ndarray,
    typewell_gr_mean: float,
) -> np.ndarray:
    filled = (
        pd.Series(np.asarray(observed_gr, dtype=np.float64))
        .interpolate(limit_direction="both")
        .fillna(float(typewell_gr_mean))
        .to_numpy(np.float64)
    )
    if not np.isfinite(filled).all():
        raise ValueError("History GR fill produced non-finite values")
    return filled


def exp209_history_sigma(
    history_tvt: np.ndarray,
    history_observed_gr: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    sigma_min: float,
    sigma_max: float,
    sigma_default: float,
) -> float:
    filled = fill_history_gr(history_observed_gr, float(np.mean(typewell_gr)))
    reference = np.interp(history_tvt, typewell_tvt, typewell_gr)
    sigma = float(np.std(filled - reference))
    if not np.isfinite(sigma):
        sigma = float(sigma_default)
    return float(np.clip(sigma, sigma_min, sigma_max))


def gaussian_log_density(
    observed_gr: float,
    lookup_gr: np.ndarray,
    sigma: float,
    squared_z_clip: float,
) -> np.ndarray:
    z_squared = ((float(observed_gr) - lookup_gr) / float(sigma)) ** 2
    return (
        -0.5 * np.minimum(z_squared, float(squared_z_clip))
        - np.log(float(sigma))
        - 0.5 * np.log(2.0 * np.pi)
    )


def logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return float(maximum + np.log(np.exp(values - maximum).sum()))


def normalized_posterior_from_log_weights(log_weights: np.ndarray) -> np.ndarray:
    normalizer = logsumexp(np.asarray(log_weights, dtype=np.float64))
    posterior = np.exp(np.asarray(log_weights, dtype=np.float64) - normalizer)
    if not np.isfinite(posterior).all() or not np.isclose(posterior.sum(), 1.0):
        raise RuntimeError("Registration posterior normalization failed")
    return posterior


def filter_registration_history(
    history_tvt: np.ndarray,
    history_observed_gr: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    offsets_ft: np.ndarray,
    transition: np.ndarray,
    initial_probability: np.ndarray,
    sigma: float,
    squared_z_clip: float,
) -> np.ndarray:
    posterior = np.asarray(initial_probability, dtype=np.float64).copy()
    filled_gr = fill_history_gr(history_observed_gr, float(np.mean(typewell_gr)))
    for row_index, (position, observed) in enumerate(
        zip(history_tvt, filled_gr, strict=True)
    ):
        if row_index:
            posterior = posterior @ transition
        lookup = np.interp(position + offsets_ft, typewell_tvt, typewell_gr)
        posterior = normalized_posterior_from_log_weights(
            np.log(np.maximum(posterior, np.finfo(np.float64).tiny))
            + gaussian_log_density(observed, lookup, sigma, squared_z_clip)
        )
    return posterior


def score_predictive_heldout(
    heldout_tvt: np.ndarray,
    heldout_observed_gr: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    offsets_ft: np.ndarray,
    transition: np.ndarray,
    history_posterior: np.ndarray,
    sigma: float,
    squared_z_clip: float,
) -> dict[str, Any]:
    posterior = np.asarray(history_posterior, dtype=np.float64).copy()
    model_nll = 0.0
    delta_zero_nll = 0.0
    observed_rows = 0
    zero_index = int(np.flatnonzero(offsets_ft == 0.0)[0])
    for position, observed in zip(
        heldout_tvt,
        heldout_observed_gr,
        strict=True,
    ):
        predictive = posterior @ transition
        if not np.isfinite(observed):
            posterior = predictive
            continue
        lookup = np.interp(position + offsets_ft, typewell_tvt, typewell_gr)
        log_density = gaussian_log_density(
            float(observed),
            lookup,
            sigma,
            squared_z_clip,
        )
        log_predictive = np.log(
            np.maximum(predictive, np.finfo(np.float64).tiny)
        )
        model_nll -= logsumexp(log_predictive + log_density)
        delta_zero_nll -= float(log_density[zero_index])
        posterior = normalized_posterior_from_log_weights(
            log_predictive + log_density
        )
        observed_rows += 1
    return {
        "model_nll": float(model_nll),
        "delta_zero_nll": float(delta_zero_nll),
        "observed_rows": int(observed_rows),
        "final_posterior": posterior,
    }


def posterior_summary(
    posterior: np.ndarray,
    offsets_ft: np.ndarray,
) -> dict[str, float | int]:
    values = np.asarray(offsets_ft, dtype=np.float64)
    probability = np.asarray(posterior, dtype=np.float64)
    mean_delta = float(probability @ values)
    return {
        "posterior_mean_delta_ft": mean_delta,
        "posterior_sign": int(np.sign(mean_delta)),
        "nonzero_posterior_mass": float(probability[values != 0.0].sum()),
        "boundary_posterior_mass": float(probability[[0, -1]].sum()),
    }


def exp209_position_grid_bounds(
    last_known_tvt: float,
    typewell_tvt: np.ndarray,
    band_pad_ft: float,
    typewell_outer_pad_ft: float,
    step_ft: float,
) -> tuple[float, float, int]:
    grid_min = max(
        float(typewell_tvt.min()) - typewell_outer_pad_ft,
        last_known_tvt - band_pad_ft,
    )
    requested_max = min(
        float(typewell_tvt.max()) + typewell_outer_pad_ft,
        last_known_tvt + band_pad_ft,
    )
    grid = np.arange(grid_min, requested_max + step_ft, step_ft, dtype=np.float64)
    return float(grid[0]), float(grid[-1]), int(len(grid))


# %% [markdown]
# ## 5. Rolling-origin Stage 0 generation

# %%
def build_well_stage0(
    well_id: str,
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    stage0 = get_nested(config, "validation.stage_0", {})
    offset = get_nested(config, "model.gr_registration_offset", {})
    fixed = get_nested(config, "model.fixed_from_exp209", {})
    known_mask = horizontal["TVT_input"].notna().to_numpy()
    known_indices = np.flatnonzero(known_mask)
    suffix_rows = int((~known_mask).sum())
    if len(known_indices) == 0:
        raise ValueError(f"No visible TVT_input prefix for well={well_id}")
    known = horizontal.iloc[known_indices].copy().reset_index(drop=True)
    known_tvt = pd.to_numeric(known["TVT_input"], errors="coerce").to_numpy(
        np.float64
    )
    known_gr = pd.to_numeric(known["GR"], errors="coerce").to_numpy(np.float64)
    if not np.isfinite(known_tvt).all():
        raise ValueError(f"Non-finite visible TVT_input for well={well_id}")
    typewell_tvt, typewell_gr = prepare_typewell(typewell)
    offsets_ft = np.asarray(offset["values_ft"], dtype=np.float64)
    initial = np.asarray(offset["initial_probability"], dtype=np.float64)
    transition = np.asarray(offset["transition_matrix"], dtype=np.float64)
    circular_gr, circular_shift = circular_shift_observed_values(
        known_gr,
        int(stage0["negative_control"]["shift_observed_values"]),
    )
    windows = fixed_rolling_windows(
        len(known),
        int(stage0["history_rows"]),
        int(stage0["heldout_rows"]),
        int(stage0["stride_rows"]),
        int(stage0["minimum_known_prefix_rows"]),
    )
    ledger_rows: list[dict[str, Any]] = []
    posterior_rows: list[dict[str, Any]] = []
    for window_id, (start, history_stop, heldout_stop) in enumerate(windows):
        history_tvt = known_tvt[start:history_stop]
        history_gr = known_gr[start:history_stop]
        heldout_tvt = known_tvt[history_stop:heldout_stop]
        heldout_gr = known_gr[history_stop:heldout_stop]
        circular_heldout_gr = circular_gr[history_stop:heldout_stop]
        history_observed = int(np.isfinite(history_gr).sum())
        heldout_observed = int(np.isfinite(heldout_gr).sum())
        circular_observed = int(np.isfinite(circular_heldout_gr).sum())
        if history_observed < int(stage0["minimum_history_observed_gr_rows"]):
            continue
        if heldout_observed < int(stage0["minimum_heldout_observed_gr_rows"]):
            continue
        if circular_observed != heldout_observed:
            raise RuntimeError("Circular control changed held-out observed-row count")
        sigma = exp209_history_sigma(
            history_tvt,
            history_gr,
            typewell_tvt,
            typewell_gr,
            float(fixed["gr_sigma_min"]),
            float(fixed["gr_sigma_max"]),
            float(fixed["gr_sigma_default"]),
        )
        history_posterior = filter_registration_history(
            history_tvt,
            history_gr,
            typewell_tvt,
            typewell_gr,
            offsets_ft,
            transition,
            initial,
            sigma,
            float(fixed["emission_squared_z_clip"]),
        )
        real = score_predictive_heldout(
            heldout_tvt,
            heldout_gr,
            typewell_tvt,
            typewell_gr,
            offsets_ft,
            transition,
            history_posterior,
            sigma,
            float(fixed["emission_squared_z_clip"]),
        )
        circular = score_predictive_heldout(
            heldout_tvt,
            circular_heldout_gr,
            typewell_tvt,
            typewell_gr,
            offsets_ft,
            transition,
            history_posterior,
            sigma,
            float(fixed["emission_squared_z_clip"]),
        )
        ledger_rows.append(
            {
                "well_id": well_id,
                "window_id": int(window_id),
                "history_start_known_row": int(start),
                "history_stop_known_row_exclusive": int(history_stop),
                "heldout_start_known_row": int(history_stop),
                "heldout_stop_known_row_exclusive": int(heldout_stop),
                "history_start_raw_row": int(known_indices[start]),
                "heldout_stop_raw_row_inclusive": int(
                    known_indices[heldout_stop - 1]
                ),
                "history_rows": int(history_stop - start),
                "heldout_rows": int(heldout_stop - history_stop),
                "history_observed_gr_rows": history_observed,
                "heldout_observed_gr_rows": heldout_observed,
                "circular_observed_gr_rows": circular_observed,
                "circular_shift_observed_values": int(circular_shift),
                "gr_sigma": float(sigma),
                "real_model_nll": float(real["model_nll"]),
                "real_delta_zero_nll": float(real["delta_zero_nll"]),
                "circular_model_nll": float(circular["model_nll"]),
                "circular_delta_zero_nll": float(circular["delta_zero_nll"]),
            }
        )
        history_summary = posterior_summary(history_posterior, offsets_ft)
        real_final_summary = posterior_summary(real["final_posterior"], offsets_ft)
        row: dict[str, Any] = {
            "well_id": well_id,
            "window_id": int(window_id),
            **history_summary,
            "final_posterior_mean_delta_ft": real_final_summary[
                "posterior_mean_delta_ft"
            ],
            "final_nonzero_posterior_mass": real_final_summary[
                "nonzero_posterior_mass"
            ],
            "final_boundary_posterior_mass": real_final_summary[
                "boundary_posterior_mass"
            ],
        }
        for state_index, value in enumerate(offsets_ft):
            label = str(int(value)).replace("-", "m")
            row[f"history_posterior_delta_{label}"] = float(
                history_posterior[state_index]
            )
            row[f"final_posterior_delta_{label}"] = float(
                real["final_posterior"][state_index]
            )
        posterior_rows.append(row)

    _, _, position_grid_count = exp209_position_grid_bounds(
        float(known_tvt[-1]),
        typewell_tvt,
        float(fixed["band_pad_ft"]),
        float(fixed["typewell_outer_pad_ft"]),
        float(fixed["position_grid_step_ft"]),
    )
    safe_horizontal = horizontal.loc[:, ["MD", "Z", "GR", "TVT_input"]]
    safe_typewell = typewell.loc[:, ["TVT", "GR"]]
    manifest = {
        "well_id": well_id,
        "status": "ok" if ledger_rows else "skipped_no_eligible_window",
        "horizontal_rows": int(len(horizontal)),
        "known_prefix_rows": int(len(known)),
        "known_prefix_observed_gr_rows": int(np.isfinite(known_gr).sum()),
        "suffix_rows": suffix_rows,
        "rolling_windows": int(len(ledger_rows)),
        "position_grid_count": position_grid_count,
        "rate_grid_count": int(fixed["n_rates"]),
        "parent_state_cell_rows": int(
            suffix_rows * position_grid_count * int(fixed["n_rates"])
        ),
        "safe_horizontal_content_sha256": dataframe_content_sha256(
            safe_horizontal
        ),
        "typewell_content_sha256": dataframe_content_sha256(safe_typewell),
    }
    return (
        pd.DataFrame(ledger_rows),
        pd.DataFrame(posterior_rows),
        manifest,
    )


# %% [markdown]
# ## 6. Five-state exact-HMM resource projection and SHA freeze

# %%
def select_resource_projection_wells(
    input_manifest: pd.DataFrame,
    count: int,
) -> pd.DataFrame:
    eligible = input_manifest.loc[
        input_manifest["parent_state_cell_rows"].notna()
        & (input_manifest["parent_state_cell_rows"] > 0)
    ].copy()
    eligible.sort_values(
        ["parent_state_cell_rows", "well_id"],
        kind="mergesort",
        inplace=True,
    )
    eligible.reset_index(drop=True, inplace=True)
    if len(eligible) < int(count):
        raise ValueError(
            f"resource projection requires {count} eligible wells, got {len(eligible)}"
        )
    positions = np.linspace(0, len(eligible) - 1, int(count), dtype=np.int64)
    if len(np.unique(positions)) != int(count):
        raise RuntimeError("Resource-projection quantile selection is not unique")
    selected = eligible.iloc[positions].copy().reset_index(drop=True)
    selected["selection_rank"] = positions.astype(np.int32)
    selected["selection_order"] = np.arange(len(selected), dtype=np.int16)
    return selected


def build_resource_projection(
    input_manifest: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    stage0 = get_nested(config, "validation.stage_0", {})
    spec = stage0["resource_projection"]
    selected = select_resource_projection_wells(
        input_manifest,
        int(stage0["resource_projection_wells"]),
    )
    offset_states = int(spec["offset_state_count"])
    alpha_bytes = int(spec["alpha_dtype_bytes"])
    posterior_bytes = int(spec["posterior_dtype_bytes"])
    emission_bytes = int(spec["emission_dtype_bytes"])
    workspace_planes = int(spec["workspace_state_planes"])
    fixed_overhead = float(spec["fixed_process_overhead_gb"]) * 1e9
    safety = float(spec["peak_rss_safety_factor"])
    selected["candidate_state_cell_rows"] = (
        selected["parent_state_cell_rows"].astype(np.int64) * offset_states
    )
    selected["alpha_tensor_bytes"] = (
        selected["candidate_state_cell_rows"].astype(np.int64) * alpha_bytes
    )
    selected["posterior_position_bytes"] = (
        selected["suffix_rows"].astype(np.int64)
        * selected["position_grid_count"].astype(np.int64)
        * posterior_bytes
    )
    selected["emission_bytes"] = (
        selected["suffix_rows"].astype(np.int64)
        * selected["position_grid_count"].astype(np.int64)
        * offset_states
        * emission_bytes
    )
    selected["workspace_bytes"] = (
        selected["position_grid_count"].astype(np.int64)
        * selected["rate_grid_count"].astype(np.int64)
        * offset_states
        * alpha_bytes
        * workspace_planes
    )
    selected["projected_peak_rss_gb"] = (
        (
            selected["alpha_tensor_bytes"]
            + selected["posterior_position_bytes"]
            + selected["emission_bytes"]
            + selected["workspace_bytes"]
            + fixed_overhead
        )
        * safety
        / 1e9
    )
    parent_runtime = float(spec["parent_reference_hmm_runtime_seconds"])
    state_multiplier = float(spec["runtime_state_count_multiplier"])
    summary = {
        "method": str(spec["method"]),
        "selection": str(spec["selection"]),
        "resource_projection_wells": int(len(selected)),
        "includes_minimum_workload": bool(
            selected["parent_state_cell_rows"].min()
            == input_manifest["parent_state_cell_rows"].dropna().min()
        ),
        "includes_maximum_workload": bool(
            selected["parent_state_cell_rows"].max()
            == input_manifest["parent_state_cell_rows"].dropna().max()
        ),
        "parent_reference_hmm_runtime_seconds": parent_runtime,
        "runtime_state_count_multiplier": state_multiplier,
        "projected_runtime_seconds": parent_runtime * state_multiplier,
        "projected_peak_rss_gb": float(selected["projected_peak_rss_gb"].max()),
        "offset_state_count": offset_states,
        "scientific_hmm_well_runs": 0,
    }
    return selected, summary


def generate_and_freeze_stage0(
    train_dir: Path,
    artifacts_dir: Path,
    config: Mapping[str, Any],
    maximum_wells: int | None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, Any],
    dict[str, Path],
]:
    wells = discover_wells(train_dir, maximum_wells)
    ledgers: list[pd.DataFrame] = []
    posteriors: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    for index, well_id in enumerate(wells, start=1):
        horizontal, typewell = load_visible_prefix_well(well_id, train_dir)
        ledger, posterior, manifest = build_well_stage0(
            well_id,
            horizontal,
            typewell,
            config,
        )
        manifests.append(manifest)
        if not ledger.empty:
            ledgers.append(ledger)
            posteriors.append(posterior)
        if index % 50 == 0 or index == len(wells):
            print(
                f"[{index}/{len(wells)}] visible-prefix registration "
                f"windows={sum(len(frame) for frame in ledgers)}"
            )
    if not ledgers:
        raise RuntimeError("No eligible Stage 0 rolling windows were generated")
    ledger = (
        pd.concat(ledgers, ignore_index=True)
        .sort_values(["well_id", "window_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    posterior = (
        pd.concat(posteriors, ignore_index=True)
        .sort_values(["well_id", "window_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    input_manifest = (
        pd.DataFrame(manifests)
        .sort_values("well_id", kind="mergesort")
        .reset_index(drop=True)
    )
    if not ledger[["well_id", "window_id"]].equals(
        posterior[["well_id", "window_id"]]
    ):
        raise RuntimeError("Rolling ledger and posterior identity mismatch")
    resource, resource_summary = build_resource_projection(input_manifest, config)
    paths = {
        "scientific_contract": artifacts_dir
        / f"{OUTPUT_PREFIX}_scientific_contract.json",
        "input_manifest": artifacts_dir / f"{OUTPUT_PREFIX}_input_manifest.csv",
        "rolling_window_ledger": artifacts_dir
        / f"{OUTPUT_PREFIX}_rolling_window_ledger.csv.gz",
        "delta_posterior": artifacts_dir
        / f"{OUTPUT_PREFIX}_delta_posterior.csv.gz",
        "resource_projection": artifacts_dir
        / f"{OUTPUT_PREFIX}_resource_projection.csv",
        "freeze_manifest": artifacts_dir / f"{OUTPUT_PREFIX}_freeze_manifest.json",
    }
    contract = stage0_contract(config)
    write_json(paths["scientific_contract"], contract)
    input_manifest.to_csv(paths["input_manifest"], index=False)
    write_deterministic_csv_gzip(ledger, paths["rolling_window_ledger"])
    write_deterministic_csv_gzip(posterior, paths["delta_posterior"])
    resource.to_csv(paths["resource_projection"], index=False)
    reports = {
        name: artifact_sha_report(path)
        for name, path in paths.items()
        if name != "freeze_manifest"
    }
    freeze = {
        "experiment": EXPERIMENT_NAME,
        "frozen_before_suffix_truth": True,
        "suffix_truth_columns_read": 0,
        "physical_prediction_rows": 0,
        "exact_hmm_well_runs": 0,
        "rolling_windows": int(len(ledger)),
        "wells": int(ledger["well_id"].nunique()),
        "contract_sha256": contract["contract_sha256"],
        "rolling_window_ledger_content_sha256": reports[
            "rolling_window_ledger"
        ]["content_sha256"],
        "delta_posterior_content_sha256": reports["delta_posterior"][
            "content_sha256"
        ],
        "resource_projection": resource_summary,
        "reports": reports,
    }
    write_json(paths["freeze_manifest"], freeze)
    for name, report in reports.items():
        if artifact_sha_report(paths[name])["content_sha256"] != report["content_sha256"]:
            raise RuntimeError(f"SHA readback failed for {name}")
    return ledger, posterior, input_manifest, resource, freeze, paths


# %% [markdown]
# ## 7. Fold metrics and promotion gates

# %%
def assign_group_folds(
    ledger: pd.DataFrame,
    posterior: pd.DataFrame,
    requested_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = ledger[["well_id", "window_id"]].copy()
    splits = min(int(requested_splits), int(merged["well_id"].nunique()))
    if splits < 2:
        merged["fold"] = 0
    else:
        fold_values = np.full(len(merged), -1, dtype=np.int16)
        groups = merged["well_id"].astype(str).to_numpy()
        splitter = GroupKFold(n_splits=splits)
        for fold, (_, valid_index) in enumerate(
            splitter.split(
                np.zeros((len(merged), 1), dtype=np.float32),
                groups=groups,
            )
        ):
            fold_values[valid_index] = int(fold)
        if np.any(fold_values < 0):
            raise RuntimeError("GroupKFold left unassigned windows")
        merged["fold"] = fold_values
    return (
        ledger.merge(
            merged,
            on=["well_id", "window_id"],
            how="left",
            validate="one_to_one",
        ),
        posterior.merge(
            merged,
            on=["well_id", "window_id"],
            how="left",
            validate="one_to_one",
        ),
    )


def adjacent_window_sign_agreement(posterior: pd.DataFrame) -> tuple[float, int]:
    agreements: list[bool] = []
    for _, frame in posterior.groupby("well_id", sort=True):
        signs = (
            frame.sort_values("window_id", kind="mergesort")["posterior_sign"]
            .to_numpy(np.int8)
        )
        if len(signs) >= 2:
            agreements.extend((signs[1:] == signs[:-1]).tolist())
    if not agreements:
        return 0.0, 0
    return float(np.mean(agreements)), int(len(agreements))


def probabilities_in_unit_interval(
    frame: pd.DataFrame,
    columns: Sequence[str],
    atol: float = 1.0e-12,
) -> bool:
    values = frame.loc[:, list(columns)].to_numpy(np.float64)
    return bool(
        np.isfinite(values).all()
        and np.all(values >= -float(atol))
        and np.all(values <= 1.0 + float(atol))
    )


def aggregate_stage0_metrics(
    ledger: pd.DataFrame,
    posterior: pd.DataFrame,
) -> dict[str, Any]:
    real_model_nll = float(ledger["real_model_nll"].sum())
    real_zero_nll = float(ledger["real_delta_zero_nll"].sum())
    circular_model_nll = float(ledger["circular_model_nll"].sum())
    circular_zero_nll = float(ledger["circular_delta_zero_nll"].sum())
    real_gain = (real_zero_nll - real_model_nll) / max(
        abs(real_zero_nll),
        np.finfo(np.float64).tiny,
    )
    circular_gain = (circular_zero_nll - circular_model_nll) / max(
        abs(circular_zero_nll),
        np.finfo(np.float64).tiny,
    )
    sign_agreement, adjacent_pairs = adjacent_window_sign_agreement(posterior)
    return {
        "windows": int(len(ledger)),
        "wells": int(ledger["well_id"].nunique()),
        "observed_heldout_rows": int(ledger["heldout_observed_gr_rows"].sum()),
        "real_model_nll": real_model_nll,
        "real_delta_zero_nll": real_zero_nll,
        "real_predictive_nll_gain_fraction": float(real_gain),
        "circular_model_nll": circular_model_nll,
        "circular_delta_zero_nll": circular_zero_nll,
        "circular_predictive_nll_gain_fraction": float(circular_gain),
        "real_minus_circular_nll_gain_fraction": float(real_gain - circular_gain),
        "nonzero_posterior_mean": float(
            posterior["nonzero_posterior_mass"].mean()
        ),
        "boundary_posterior_mean": float(
            posterior["boundary_posterior_mass"].mean()
        ),
        "adjacent_window_sign_agreement": sign_agreement,
        "adjacent_window_pairs": adjacent_pairs,
    }


def build_fold_metrics(
    ledger: pd.DataFrame,
    posterior: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    gates = get_nested(config, "validation.stage_0.all_required", {})
    records: list[dict[str, Any]] = []
    for fold, fold_ledger in ledger.groupby("fold", sort=True):
        fold_posterior = posterior.loc[posterior["fold"].eq(fold)]
        metrics = aggregate_stage0_metrics(fold_ledger, fold_posterior)
        direction_pass = bool(
            metrics["real_predictive_nll_gain_fraction"]
            >= float(gates["minimum_predictive_gr_nll_gain_fraction"])
            and metrics["real_minus_circular_nll_gain_fraction"]
            >= float(gates["minimum_real_minus_circular_nll_gain_fraction"])
        )
        records.append(
            {"fold": int(fold), **metrics, "direction_pass": direction_pass}
        )
    return pd.DataFrame(records)


def evaluate_stage0_gates(
    ledger: pd.DataFrame,
    posterior: pd.DataFrame,
    input_manifest: pd.DataFrame,
    resource: pd.DataFrame,
    freeze: Mapping[str, Any],
    fold_metrics: pd.DataFrame,
    config: Mapping[str, Any],
    debug: bool,
) -> dict[str, Any]:
    gates = get_nested(config, "validation.stage_0.all_required", {})
    overall = aggregate_stage0_metrics(ledger, posterior)
    nonzero_low, nonzero_high = gates["nonzero_posterior_mean_range"]
    expected_folds = list(get_nested(config, "validation.expected_folds"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    resource_summary = freeze["resource_projection"]
    technical = {
        "suffix_truth_columns_read_zero": (
            int(freeze["suffix_truth_columns_read"]) == 0
        ),
        "physical_prediction_rows_zero": (
            int(freeze["physical_prediction_rows"]) == 0
        ),
        "exact_hmm_well_runs_zero": int(freeze["exact_hmm_well_runs"]) == 0,
        "rolling_keys_unique": not ledger.duplicated(
            ["well_id", "window_id"]
        ).any(),
        "posterior_keys_unique": not posterior.duplicated(
            ["well_id", "window_id"]
        ).any(),
        "all_nll_finite": bool(
            np.isfinite(
                ledger[
                    [
                        "real_model_nll",
                        "real_delta_zero_nll",
                        "circular_model_nll",
                        "circular_delta_zero_nll",
                    ]
                ].to_numpy(np.float64)
            ).all()
        ),
        "posterior_in_unit_interval": probabilities_in_unit_interval(
            posterior,
            ["nonzero_posterior_mass", "boundary_posterior_mass"],
        ),
        "observed_folds_exact": sorted(fold_metrics["fold"].tolist())
        == expected_folds,
        "expected_wells_exact": debug
        or int(input_manifest["well_id"].nunique()) == expected_wells,
        "scored_wells_are_input_wells": set(ledger["well_id"].astype(str)).issubset(
            set(input_manifest["well_id"].astype(str))
        ),
        "all_expected_wells_have_windows": debug
        or int(ledger["well_id"].nunique()) == expected_wells,
        "resource_projection_has_16_wells": len(resource) == 16,
        "resource_projection_includes_extrema": bool(
            resource_summary["includes_minimum_workload"]
            and resource_summary["includes_maximum_workload"]
        ),
        "stage1_disabled": not bool(get_nested(config, "execution.run_stage_1")),
    }
    scientific = {
        "predictive_nll_gain": (
            overall["real_predictive_nll_gain_fraction"]
            >= float(gates["minimum_predictive_gr_nll_gain_fraction"])
        ),
        "minimum_passing_folds": (
            int(fold_metrics["direction_pass"].sum())
            >= int(gates["minimum_passing_folds"])
        ),
        "real_minus_circular_nll_gain": (
            overall["real_minus_circular_nll_gain_fraction"]
            >= float(gates["minimum_real_minus_circular_nll_gain_fraction"])
        ),
        "nonzero_posterior_mass_non_degenerate": (
            float(nonzero_low)
            <= overall["nonzero_posterior_mean"]
            <= float(nonzero_high)
        ),
        "boundary_posterior_mass_bounded": (
            overall["boundary_posterior_mean"]
            <= float(gates["maximum_boundary_posterior_mass"])
        ),
        "adjacent_window_sign_stable": (
            overall["adjacent_window_sign_agreement"]
            >= float(gates["minimum_adjacent_window_sign_agreement"])
        ),
        "projected_runtime_within_budget": (
            float(resource_summary["projected_runtime_seconds"])
            <= float(gates["maximum_projected_runtime_seconds"])
        ),
        "projected_peak_rss_within_budget": (
            float(resource_summary["projected_peak_rss_gb"])
            <= float(gates["maximum_projected_peak_rss_gb"])
        ),
    }
    technical_pass = bool(all(technical.values()))
    scientific_pass = bool(all(scientific.values()))
    passed = bool(not debug and technical_pass and scientific_pass)
    return {
        "technical": technical,
        "scientific": scientific,
        "technical_pass": technical_pass,
        "scientific_pass": scientific_pass,
        "stage0_pass": passed,
        "stage1_eligible": passed,
        "debug_never_promotes": bool(debug),
        "overall": overall,
        "passing_folds": int(fold_metrics["direction_pass"].sum()),
        "resource": resource_summary,
        "decision_if_run": (
            "STAGE0_PASS_REQUEST_SEPARATE_STAGE1_APPROVAL"
            if passed
            else "STAGE0_FAIL_CLOSE_WITHOUT_RESCUE"
            if not debug
            else "DEBUG_ONLY_NO_DECISION"
        ),
    }


# %% [markdown]
# ## 8. Execution orchestration and generated artifacts

# %%
def run_stage0(
    config: Mapping[str, Any],
    package_dir: Path,
    maximum_wells: int | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    started = time.time()
    require_authoritative_runtime()
    contract = validate_scientific_contract(
        config,
        require_run_approval=True,
    )
    train_dir = resolve_train_dir(config)
    artifacts_dir, _ = output_dirs(package_dir)
    print("Experiment:", EXPERIMENT_NAME)
    print("Route:", get_nested(config, "experiment.route"))
    print("Parent:", get_nested(config, "lineage.parent"))
    print("Contract SHA256:", contract["contract_sha256"])
    print("Stage 0: 1 diagnostic / 5 offsets / 5 folds / 16 resource wells")
    print("Exact HMM / LightGBM / boosters / control reruns: 0 / 0 / 0 / 0")
    print("Stage 1 planned: 1 variant / 773 exact-HMM wells (disabled)")

    ledger, posterior, input_manifest, resource, freeze, frozen_paths = (
        generate_and_freeze_stage0(
            train_dir,
            artifacts_dir,
            config,
            maximum_wells,
        )
    )
    ledger, posterior = assign_group_folds(
        ledger,
        posterior,
        int(get_nested(config, "validation.n_folds")),
    )
    fold_metrics = build_fold_metrics(ledger, posterior, config)
    gate_report = evaluate_stage0_gates(
        ledger,
        posterior,
        input_manifest,
        resource,
        freeze,
        fold_metrics,
        config,
        debug,
    )
    paths = {
        "fold_metrics": artifacts_dir / f"{OUTPUT_PREFIX}_fold_metrics.csv",
        "gate_report": artifacts_dir / f"{OUTPUT_PREFIX}_gate_report.json",
        "summary": artifacts_dir / f"{OUTPUT_PREFIX}_summary.json",
    }
    fold_metrics.to_csv(paths["fold_metrics"], index=False)
    write_json(paths["gate_report"], gate_report)
    output_reports = {
        **{
            name: artifact_sha_report(path)
            for name, path in frozen_paths.items()
        },
        "fold_metrics": artifact_sha_report(paths["fold_metrics"]),
        "gate_report": artifact_sha_report(paths["gate_report"]),
    }
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": (
            "debug_completed"
            if debug
            else "stage0_pass"
            if gate_report["stage0_pass"]
            else "stage0_failed_close_without_rescue"
        ),
        "created_at": datetime.now(UTC).isoformat(),
        "debug": bool(debug),
        "maximum_wells": maximum_wells,
        "elapsed_seconds": float(time.time() - started),
        "contract_sha256": contract["contract_sha256"],
        "train_dir": str(train_dir),
        "input_wells": int(input_manifest["well_id"].nunique()),
        "scored_wells": int(ledger["well_id"].nunique()),
        "rolling_windows": int(len(ledger)),
        "execution_counts": get_nested(config, "execution.stage_0_counts"),
        "freeze": freeze,
        "fold_metrics": fold_metrics.to_dict(orient="records"),
        "gates": gate_report,
        "outputs": output_reports,
    }
    write_json(paths["summary"], summary)
    metrics_path = (
        KAGGLE_WORKING_ROOT / "metrics.json"
        if is_kaggle_runtime()
        else package_dir / "metrics.json"
    )
    write_json(
        metrics_path,
        {
            "experiment": EXPERIMENT_NAME,
            "route": "pf_beam",
            "status": summary["status"],
            "updated_at": datetime.now(UTC).date().isoformat(),
            "cv": None,
            "public_lb": None,
            "private_lb": None,
            "metric": "known_prefix_predictive_gr_nll",
            "stage_0": {
                "overall": gate_report["overall"],
                "passing_folds": gate_report["passing_folds"],
                "resource": gate_report["resource"],
                "gates": gate_report,
                "summary_path": str(paths["summary"]),
                "summary_sha256": sha256_file(paths["summary"]),
            },
            "notes": (
                "Stage 0 only; suffix truth, physical prediction, Stage 1 exact "
                "HMM, inference, blend, and submission remain disabled."
            ),
        },
    )
    print(fold_metrics.to_string(index=False))
    print(json.dumps(gate_report, indent=2, default=json_default))
    return summary


# %% [markdown]
# ## 9. Setup and configuration preview

# %%
CONFIG = load_config()
CONTRACT = validate_scientific_contract(CONFIG)

if EXECUTE_NOTEBOOK:
    display(
        {
            "experiment": EXPERIMENT_NAME,
            "route": get_nested(CONFIG, "experiment.route"),
            "parent": get_nested(CONFIG, "lineage.parent"),
            "status": get_nested(CONFIG, "experiment.status"),
            "contract_sha256": CONTRACT["contract_sha256"],
            "offset_values_ft": CONTRACT["offset_values_ft"],
            "history_heldout_stride": [
                CONTRACT["history_rows"],
                CONTRACT["heldout_rows"],
                CONTRACT["stride_rows"],
            ],
            "stage_0_counts": get_nested(CONFIG, "execution.stage_0_counts"),
            "run_stage_0": get_nested(CONFIG, "execution.run_stage_0"),
            "stage_1_implemented": False,
            "run_stage_1": False,
            "inference_enabled": False,
            "submission_enabled": False,
        }
    )


# %% [markdown]
# ## 10. Fail-closed Stage 0 execution selection

# %%
if EXECUTE_NOTEBOOK:
    if bool(get_nested(CONFIG, "execution.run_stage_0")):
        DEBUG = os.environ.get("EXPERIMENT_DEBUG", "0") == "1"
        MAX_WELLS_ENV = os.environ.get("EXPERIMENT_MAX_WELLS")
        MAX_WELLS = int(MAX_WELLS_ENV) if MAX_WELLS_ENV else None
        STAGE0_SUMMARY = run_stage0(
            CONFIG,
            package_dir=resolve_package_dir(),
            maximum_wells=MAX_WELLS,
            debug=DEBUG,
        )
    else:
        print(
            "exp365 Stage 0 is implemented, but package/push/run approval is "
            "false. No diagnostic, exact HMM, inference, or submission ran."
        )
