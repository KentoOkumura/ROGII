# %% [markdown]
# # exp367_stratified_signed_curvature_pf train
#
# Stage 0 only: generate three deterministic signed-curvature paths with the
# unchanged exp072 observation/dynamics contract, freeze paths and GR scores
# before reading suffix truth, and evaluate the frozen ranking. This notebook
# does not run a particle filter, train a model, or create a submission.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime, configuration, path, and SHA helpers
# 3. Frozen Stage 0 scientific contract
# 4. Truth-free raw input and signed-path helpers
# 5. Truth-free candidate generation and SHA freeze
# 6. Late truth and hidden-like attachment
# 7. Stage 0 metrics and gates
# 8. Execution orchestration and generated artifacts

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold

EXPERIMENT_NAME = "exp367_stratified_signed_curvature_pf"
OUTPUT_PREFIX = f"{EXPERIMENT_NAME}_stage0"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
SIGN_COLUMN = {-1: "minus", 0: "zero", 1: "plus"}
FORBIDDEN_PREFREEZE_HORIZONTAL_COLUMNS = {
    "TVT",
    "true_tvt",
    "target",
    "abs_error",
    "oracle_sign",
}


# %% [markdown]
# ## 2. Runtime, configuration, path, and SHA helpers
#
# The notebook is self-contained and notebook-safe: it does not rely on a
# module-file global. Local execution remains disabled unless explicitly opted in.

# %%
def get_nested(mapping: dict[str, Any], dotted_key: str) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
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
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


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
    schema = [
        {"name": str(column), "dtype": str(frame[column].dtype)}
        for column in frame.columns
    ]
    return mapping_sha256(schema)


def stable_seed(
    experiment: str,
    well: str,
    family: str,
    seed_index: int,
    modulo: int = 2_147_483_647,
) -> int:
    key = f"{experiment}|{well}|{family}|{int(seed_index)}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % modulo + 1


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=json_default) + "\n"
    )


def write_deterministic_csv_gzip(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        float_format="%.10g",
    )


def candidate_package_dirs() -> list[Path]:
    cwd = Path.cwd()
    candidates = [
        cwd,
        cwd / "experiments" / EXPERIMENT_NAME,
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
        config_path = candidate / "config.yaml"
        if not config_path.exists():
            continue
        try:
            loaded = yaml.safe_load(config_path.read_text()) or {}
        except (OSError, yaml.YAMLError):
            continue
        if get_nested(loaded, "experiment.name") == EXPERIMENT_NAME:
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
        "exp367 Stage 0 must run in Kaggle. "
        "Set EXPERIMENT_ALLOW_LOCAL=1 only for an explicitly approved local smoke run."
    )


def resolve_train_dir(config: dict[str, Any]) -> Path:
    configured = Path(str(get_nested(config, "data.train_dir") or "data/raw/train"))
    candidates = [
        configured,
        Path.cwd() / configured,
        Path.cwd() / "data" / "raw" / "train",
    ]
    if KAGGLE_INPUT_ROOT.exists():
        competition_slug = "rogii-wellbore-geology-prediction"
        candidates.extend(
            [
                KAGGLE_INPUT_ROOT / competition_slug / "train",
                *sorted(KAGGLE_INPUT_ROOT.glob("**/train")),
            ]
        )
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if any(candidate.glob("*__horizontal_well.csv")):
            return candidate
    raise FileNotFoundError("Could not resolve raw train directory")


def output_dirs(package_dir: Path) -> tuple[Path, Path]:
    if is_kaggle_runtime():
        root = KAGGLE_WORKING_ROOT
    else:
        root = package_dir
    artifacts = root / "artifacts"
    features = root / "features"
    artifacts.mkdir(parents=True, exist_ok=True)
    features.mkdir(parents=True, exist_ok=True)
    return artifacts, features


def resolve_existing_file(
    candidates: Sequence[str | Path],
    patterns: Sequence[str],
    filename: str,
) -> Path:
    roots = [Path.cwd(), resolve_package_dir()]
    for raw in candidates:
        path = Path(raw)
        options = [path] if path.is_absolute() else [path, *(root / path for root in roots)]
        for option in options:
            candidate = option / filename if option.is_dir() else option
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
    if KAGGLE_INPUT_ROOT.exists():
        for pattern in patterns:
            for candidate in sorted(KAGGLE_INPUT_ROOT.glob(pattern)):
                if candidate.is_file() and candidate.name == filename:
                    return candidate
    raise FileNotFoundError(f"Could not resolve required file: {filename}")


# %% [markdown]
# ## 3. Frozen Stage 0 scientific contract
#
# This validates the design before any raw data are read. Stage 1 remains
# disabled and is deliberately absent from this notebook.

# %%
def stage0_contract(config: dict[str, Any]) -> dict[str, Any]:
    stage0 = get_nested(config, "validation.stage_0") or {}
    signed = get_nested(config, "model.signed_curvature") or {}
    fixed = get_nested(config, "model.fixed_from_exp072") or {}
    execution = get_nested(config, "execution") or {}
    contract = {
        "experiment": get_nested(config, "experiment.name"),
        "route": get_nested(config, "experiment.route"),
        "implementation_scope": execution.get("implementation_scope"),
        "run_stage_1": execution.get("run_stage_1"),
        "candidate_signs": stage0.get("candidate_signs"),
        "stable_score_tie_break": stage0.get("stable_score_tie_break"),
        "zero_first_reference_rank": stage0.get("zero_first_reference_rank"),
        "block_rows": stage0.get("block_rows"),
        "stride_rows": stage0.get("stride_rows"),
        "incomplete_tail_policy": stage0.get("incomplete_tail_policy"),
        "negative_control": stage0.get("negative_control"),
        "negative_control_shift_blocks": stage0.get(
            "negative_control_shift_blocks"
        ),
        "truth_label_after_freeze": stage0.get("truth_label_after_freeze"),
        "positive_scope_direction_metric": stage0.get(
            "positive_scope_direction_metric"
        ),
        "curvature_states": signed.get("states"),
        "rate_drift_per_row": signed.get("rate_drift_per_row"),
        "transition_matrix": signed.get("transition_matrix"),
        "initial_particle_counts": signed.get("initial_particle_counts"),
        "minimum_post_resample_count_each_sign": signed.get(
            "minimum_post_resample_count_each_sign"
        ),
        "particles": fixed.get("particles"),
        "seed_count": fixed.get("seed_count"),
        "momentum": fixed.get("momentum"),
        "velocity_noise": fixed.get("velocity_noise"),
        "position_noise": fixed.get("position_noise"),
        "resample_threshold": fixed.get("resample_threshold"),
        "resample_position_noise": fixed.get("resample_position_noise"),
        "resample_velocity_noise": fixed.get("resample_velocity_noise"),
        "typewell_grid_step_ft": fixed.get("typewell_grid_step_ft"),
        "typewell_support_pad_ft": fixed.get("typewell_support_pad_ft"),
        "gr_sigma_min": fixed.get("gr_sigma_min"),
        "gr_sigma_max": fixed.get("gr_sigma_max"),
        "gr_sigma_default": fixed.get("gr_sigma_default"),
        "initial_rate_window_rows": fixed.get("initial_rate_window_rows"),
        "initial_rate_min_valid_steps": fixed.get("initial_rate_min_valid_steps"),
        "initial_rate_fallback": fixed.get("initial_rate_fallback"),
        "initial_rate_source": fixed.get("initial_rate_source"),
        "pf_seed_well_runs": stage0.get("pf_seed_well_runs"),
    }
    contract["contract_sha256"] = mapping_sha256(contract)
    return contract


def validate_scientific_contract(config: dict[str, Any]) -> dict[str, Any]:
    contract = stage0_contract(config)
    required_equal = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "implementation_scope": "stage0_only",
        "run_stage_1": False,
        "candidate_signs": [-1, 0, 1],
        "stable_score_tie_break": [0, -1, 1],
        "zero_first_reference_rank": [0, -1, 1],
        "block_rows": 512,
        "stride_rows": 256,
        "incomplete_tail_policy": "exclude",
        "negative_control": "within_well_circular_shift_of_gr_blocks",
        "negative_control_shift_blocks": 1,
        "truth_label_after_freeze": "nearest_path_by_block_tvt_rmse",
        "positive_scope_direction_metric": (
            "selected_path_rmse_gain_vs_zero_path_ft"
        ),
        "curvature_states": [-1, 0, 1],
        "rate_drift_per_row": 0.000009765625,
        "initial_particle_counts": [100, 300, 100],
        "minimum_post_resample_count_each_sign": 50,
        "particles": 500,
        "seed_count": 128,
        "momentum": 0.998,
        "velocity_noise": 0.002,
        "position_noise": 0.005,
        "resample_threshold": 0.5,
        "resample_position_noise": 0.10,
        "resample_velocity_noise": 0.001,
        "typewell_grid_step_ft": 0.2,
        "typewell_support_pad_ft": 100.0,
        "gr_sigma_min": 10.0,
        "gr_sigma_max": 60.0,
        "gr_sigma_default": 30.0,
        "initial_rate_window_rows": 30,
        "initial_rate_min_valid_steps": 3,
        "initial_rate_fallback": 0.0,
        "initial_rate_source": (
            "unchanged_exp072_terminal_difference_heuristic_no_regressor"
        ),
        "pf_seed_well_runs": 0,
    }
    mismatches = {
        key: {"expected": expected, "actual": contract.get(key)}
        for key, expected in required_equal.items()
        if contract.get(key) != expected
    }
    transition = np.asarray(contract["transition_matrix"], dtype=np.float64)
    if transition.shape != (3, 3):
        mismatches["transition_matrix_shape"] = {
            "expected": [3, 3],
            "actual": list(transition.shape),
        }
    elif not np.allclose(transition.sum(axis=1), 1.0, atol=1e-12, rtol=0.0):
        mismatches["transition_matrix_row_sums"] = {
            "expected": [1.0, 1.0, 1.0],
            "actual": transition.sum(axis=1).tolist(),
        }
    if mismatches:
        raise ValueError(f"Frozen exp367 Stage 0 contract mismatch: {mismatches}")
    if not bool(get_nested(config, "execution.implementation_approved")):
        raise ValueError("Stage 0 implementation approval must be recorded")
    return contract


@dataclass
class TruthAccessLedger:
    frozen: bool = False
    truth_rows_before_freeze: int = 0
    hidden_role_rows_before_freeze: int = 0
    truth_rows_after_freeze: int = 0
    hidden_role_rows_after_freeze: int = 0

    def guard_prefreeze_columns(
        self,
        columns: Iterable[str],
        rows: int,
        label: str,
    ) -> None:
        overlap = FORBIDDEN_PREFREEZE_HORIZONTAL_COLUMNS.intersection(columns)
        if overlap:
            if not self.frozen:
                self.truth_rows_before_freeze += int(rows)
            raise ValueError(f"{label}: forbidden pre-freeze columns requested: {sorted(overlap)}")

    def freeze(self) -> None:
        if self.truth_rows_before_freeze or self.hidden_role_rows_before_freeze:
            raise RuntimeError("truth or hidden-like roles were accessed before SHA freeze")
        self.frozen = True

    def record_truth_late(self, rows: int) -> None:
        if not self.frozen:
            self.truth_rows_before_freeze += int(rows)
            raise RuntimeError("suffix truth cannot be read before candidate SHA freeze")
        self.truth_rows_after_freeze += int(rows)

    def record_hidden_late(self, rows: int) -> None:
        if not self.frozen:
            self.hidden_role_rows_before_freeze += int(rows)
            raise RuntimeError("hidden-like roles cannot be read before candidate SHA freeze")
        self.hidden_role_rows_after_freeze += int(rows)


# %% [markdown]
# ## 4. Truth-free raw input and signed-path helpers
#
# Horizontal `TVT` is not requested here. The unchanged exp072 terminal
# difference heuristic is retained as fixed initialization; no learned prefix
# or geometry rate regressor is introduced.

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


def load_prefreeze_well(
    well: str,
    train_dir: Path,
    ledger: TruthAccessLedger,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    horizontal_path = train_dir / f"{well}__horizontal_well.csv"
    typewell_path = train_dir / f"{well}__typewell.csv"
    horizontal_columns = ["MD", "Z", "GR", "TVT_input"]
    ledger.guard_prefreeze_columns(horizontal_columns, 0, f"{well} horizontal")
    horizontal = pd.read_csv(horizontal_path, usecols=horizontal_columns)
    typewell = pd.read_csv(typewell_path, usecols=["TVT", "GR"])
    if horizontal.empty or typewell.empty:
        raise ValueError(f"Empty raw input for well={well}")
    return horizontal, typewell, horizontal_path, typewell_path


def interpolate_observed_gr(values: pd.Series, fallback: float) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").astype(np.float64)
    return (
        numeric.interpolate(limit_direction="both")
        .fillna(float(fallback))
        .to_numpy(np.float64)
    )


def prepare_typewell_grid(
    typewell: pd.DataFrame,
    grid_step: float,
) -> tuple[np.ndarray, float, float, np.ndarray, np.ndarray]:
    clean = typewell.copy()
    clean["TVT"] = pd.to_numeric(clean["TVT"], errors="coerce")
    clean["GR"] = pd.to_numeric(clean["GR"], errors="coerce")
    clean = (
        clean.dropna(subset=["TVT", "GR"])
        .sort_values("TVT")
        .drop_duplicates("TVT", keep="last")
    )
    if len(clean) < 3:
        raise ValueError("Typewell must contain at least three finite TVT/GR rows")
    tw_tvt = clean["TVT"].to_numpy(np.float64)
    tw_gr = clean["GR"].to_numpy(np.float64)
    grid_min = float(tw_tvt.min())
    grid = np.arange(grid_min, float(tw_tvt.max()) + grid_step, grid_step)
    grid_gr = np.interp(grid, tw_tvt, tw_gr).astype(np.float64)
    return grid_gr, grid_min, float(grid_step), tw_tvt, tw_gr


def interpolate_uniform_grid(
    grid_values: np.ndarray,
    values: np.ndarray,
    grid_min: float,
    grid_step: float,
) -> np.ndarray:
    coordinates = (np.asarray(values, np.float64) - grid_min) / grid_step
    left = np.floor(coordinates).astype(np.int64)
    fraction = coordinates - left
    left = np.clip(left, 0, len(grid_values) - 1)
    right = np.clip(left + 1, 0, len(grid_values) - 1)
    return grid_values[left] * (1.0 - fraction) + grid_values[right] * fraction


def exp072_gr_sigma(
    horizontal: pd.DataFrame,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    sigma_min: float,
    sigma_max: float,
    sigma_default: float,
) -> float:
    known_mask = horizontal["TVT_input"].notna() & horizontal["GR"].notna()
    known = horizontal.loc[known_mask]
    if len(known) < 20:
        return float(sigma_default)
    observed = pd.to_numeric(known["GR"], errors="coerce").to_numpy(np.float64)
    known_tvt = pd.to_numeric(
        known["TVT_input"], errors="coerce"
    ).to_numpy(np.float64)
    residual = observed - np.interp(known_tvt, tw_tvt, tw_gr)
    sigma = float(np.std(residual))
    return float(np.clip(sigma, sigma_min, sigma_max))


def exp072_initial_rate(
    known: pd.DataFrame,
    window_rows: int,
    minimum_valid_steps: int,
    fallback: float,
) -> tuple[float, int]:
    tail = known.tail(int(window_rows))
    tvt = pd.to_numeric(tail["TVT_input"], errors="coerce").to_numpy(np.float64)
    z = pd.to_numeric(tail["Z"], errors="coerce").to_numpy(np.float64)
    md = pd.to_numeric(tail["MD"], errors="coerce").to_numpy(np.float64)
    dt = np.diff(tvt)
    dz = np.diff(z)
    dm = np.diff(md)
    valid = np.isfinite(dt) & np.isfinite(dz) & np.isfinite(dm) & (dm > 0.0)
    count = int(valid.sum())
    if count < int(minimum_valid_steps):
        return float(fallback), count
    return float(np.median((dt[valid] + dz[valid]) / dm[valid])), count


def generate_fixed_signed_paths(
    eval_md: np.ndarray,
    eval_z: np.ndarray,
    last_known_md: float,
    last_known_position: float,
    initial_rate: float,
    signs: Sequence[int],
    momentum: float,
    rate_drift_per_row: float,
    typewell_min: float,
    typewell_max: float,
    support_pad: float,
) -> dict[int, np.ndarray]:
    md = np.asarray(eval_md, dtype=np.float64)
    z = np.asarray(eval_z, dtype=np.float64)
    if md.ndim != 1 or z.ndim != 1 or len(md) != len(z):
        raise ValueError("eval_md and eval_z must be aligned one-dimensional arrays")
    paths: dict[int, np.ndarray] = {}
    for sign in signs:
        position = float(last_known_position)
        rate = float(initial_rate)
        previous_md = float(last_known_md)
        path = np.empty(len(md), dtype=np.float64)
        for row_index in range(len(md)):
            delta_md = float(md[row_index] - previous_md)
            if not np.isfinite(delta_md) or delta_md < 1.0:
                delta_md = 1.0
            rate = float(momentum) * rate + float(sign) * float(
                rate_drift_per_row
            )
            position += rate * delta_md
            tvt = position - float(z[row_index])
            tvt = float(
                np.clip(
                    tvt,
                    float(typewell_min) - float(support_pad),
                    float(typewell_max) + float(support_pad),
                )
            )
            position = tvt + float(z[row_index])
            path[row_index] = tvt
            previous_md = float(md[row_index])
        paths[int(sign)] = path
    return paths


def gaussian_gr_score(
    observed_gr: np.ndarray,
    path_tvt: np.ndarray,
    grid_gr: np.ndarray,
    grid_min: float,
    grid_step: float,
    sigma: float,
) -> float:
    expected_gr = interpolate_uniform_grid(
        grid_gr,
        np.asarray(path_tvt, np.float64),
        grid_min,
        grid_step,
    )
    residual = (np.asarray(observed_gr, np.float64) - expected_gr) / float(sigma)
    squared = np.minimum(residual * residual, 600.0)
    return float(np.mean(-0.5 * squared))


def fixed_full_blocks(
    row_count: int,
    block_rows: int,
    stride_rows: int,
) -> list[tuple[int, int]]:
    if row_count < block_rows:
        return []
    return [
        (start, start + block_rows)
        for start in range(0, row_count - block_rows + 1, stride_rows)
    ]


def circular_control_gr_blocks(
    blocks: Sequence[np.ndarray],
    shift_blocks: int,
    single_block_shift_rows: int,
) -> list[np.ndarray]:
    copied = [np.asarray(block, np.float64).copy() for block in blocks]
    if not copied:
        return []
    if len(copied) == 1:
        shift = int(single_block_shift_rows) % len(copied[0])
        return [np.roll(copied[0], shift)]
    shift = int(shift_blocks) % len(copied)
    return [copied[(index + shift) % len(copied)] for index in range(len(copied))]


def ranked_signs(
    score_by_sign: dict[int, float],
    tie_break: Sequence[int],
) -> list[int]:
    priority = {int(sign): index for index, sign in enumerate(tie_break)}
    return sorted(
        (int(sign) for sign in score_by_sign),
        key=lambda sign: (-float(score_by_sign[sign]), priority[sign]),
    )


def build_prefreeze_rows_for_well(
    well: str,
    train_dir: Path,
    config: dict[str, Any],
    ledger: TruthAccessLedger,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    horizontal, typewell, horizontal_path, typewell_path = load_prefreeze_well(
        well,
        train_dir,
        ledger,
    )
    known_mask = horizontal["TVT_input"].notna().to_numpy()
    eval_mask = ~known_mask
    known_indices = np.flatnonzero(known_mask)
    eval_indices = np.flatnonzero(eval_mask)
    if len(known_indices) < 4:
        return pd.DataFrame(), pd.DataFrame(), {
            "well_id": well,
            "status": "skipped_short_prefix",
            "prefix_rows": int(len(known_indices)),
            "suffix_rows": int(len(eval_indices)),
        }
    if len(eval_indices) == 0:
        return pd.DataFrame(), pd.DataFrame(), {
            "well_id": well,
            "status": "skipped_no_suffix",
            "prefix_rows": int(len(known_indices)),
            "suffix_rows": 0,
        }
    if int(known_indices[-1]) >= int(eval_indices[0]):
        raise ValueError(f"well={well} does not have one contiguous visible prefix")

    stage0 = get_nested(config, "validation.stage_0") or {}
    signed = get_nested(config, "model.signed_curvature") or {}
    fixed = get_nested(config, "model.fixed_from_exp072") or {}
    signs = [int(value) for value in stage0["candidate_signs"]]
    grid_gr, grid_min, grid_step, tw_tvt, tw_gr = prepare_typewell_grid(
        typewell,
        float(fixed["typewell_grid_step_ft"]),
    )
    observed_gr = interpolate_observed_gr(
        horizontal["GR"],
        fallback=float(np.mean(tw_gr)),
    )
    sigma = exp072_gr_sigma(
        horizontal,
        tw_tvt,
        tw_gr,
        float(fixed["gr_sigma_min"]),
        float(fixed["gr_sigma_max"]),
        float(fixed["gr_sigma_default"]),
    )
    known = horizontal.loc[known_mask]
    initial_rate, valid_rate_steps = exp072_initial_rate(
        known,
        int(fixed["initial_rate_window_rows"]),
        int(fixed["initial_rate_min_valid_steps"]),
        float(fixed["initial_rate_fallback"]),
    )
    last_known_index = int(known_indices[-1])
    last_known = horizontal.iloc[last_known_index]
    eval_md = pd.to_numeric(
        horizontal.loc[eval_mask, "MD"], errors="coerce"
    ).to_numpy(np.float64)
    eval_z = pd.to_numeric(
        horizontal.loc[eval_mask, "Z"], errors="coerce"
    ).to_numpy(np.float64)
    eval_gr = observed_gr[eval_indices]
    if not np.isfinite(eval_md).all() or not np.isfinite(eval_z).all():
        raise ValueError(f"Non-finite MD/Z in evaluation suffix for well={well}")
    paths = generate_fixed_signed_paths(
        eval_md=eval_md,
        eval_z=eval_z,
        last_known_md=float(last_known["MD"]),
        last_known_position=float(last_known["TVT_input"]) + float(last_known["Z"]),
        initial_rate=initial_rate,
        signs=signs,
        momentum=float(fixed["momentum"]),
        rate_drift_per_row=float(signed["rate_drift_per_row"]),
        typewell_min=float(tw_tvt.min()),
        typewell_max=float(tw_tvt.max()),
        support_pad=float(fixed["typewell_support_pad_ft"]),
    )
    md_since = eval_md - float(last_known["MD"])
    candidate_rows = pd.DataFrame(
        {
            "id": [f"{well}_{int(row_index)}" for row_index in eval_indices],
            "well_id": well,
            "row_index": eval_indices.astype(np.int32),
            "suffix_row": np.arange(len(eval_indices), dtype=np.int32),
            "md_since": md_since.astype(np.float32),
            **{
                f"path_{SIGN_COLUMN[sign]}": paths[sign].astype(np.float32)
                for sign in signs
            },
        }
    )

    blocks = fixed_full_blocks(
        len(eval_indices),
        int(stage0["block_rows"]),
        int(stage0["stride_rows"]),
    )
    real_gr_blocks = [eval_gr[start:stop] for start, stop in blocks]
    circular_gr = circular_control_gr_blocks(
        real_gr_blocks,
        int(stage0["negative_control_shift_blocks"]),
        int(stage0["single_block_negative_control_shift_rows"]),
    )
    score_rows: list[dict[str, Any]] = []
    for block_index, ((start, stop), real_gr, circular) in enumerate(
        zip(blocks, real_gr_blocks, circular_gr, strict=True)
    ):
        real_score = {
            sign: gaussian_gr_score(
                real_gr,
                paths[sign][start:stop],
                grid_gr,
                grid_min,
                grid_step,
                sigma,
            )
            for sign in signs
        }
        circular_score = {
            sign: gaussian_gr_score(
                circular,
                paths[sign][start:stop],
                grid_gr,
                grid_min,
                grid_step,
                sigma,
            )
            for sign in signs
        }
        real_rank = ranked_signs(real_score, stage0["stable_score_tie_break"])
        circular_rank = ranked_signs(
            circular_score,
            stage0["stable_score_tie_break"],
        )
        row: dict[str, Any] = {
            "well_id": well,
            "block_index": int(block_index),
            "start_suffix_row": int(start),
            "stop_suffix_row_exclusive": int(stop),
            "start_row_index": int(eval_indices[start]),
            "stop_row_index_inclusive": int(eval_indices[stop - 1]),
            "start_id": f"{well}_{int(eval_indices[start])}",
            "stop_id": f"{well}_{int(eval_indices[stop - 1])}",
            "block_rows": int(stop - start),
            "block_mid_md_since": float(np.mean(md_since[start:stop])),
            "gr_sigma": float(sigma),
            "real_top1_sign": int(real_rank[0]),
            "circular_top1_sign": int(circular_rank[0]),
            "real_rank_order": ",".join(str(value) for value in real_rank),
            "circular_rank_order": ",".join(str(value) for value in circular_rank),
        }
        for sign in signs:
            suffix = SIGN_COLUMN[sign]
            row[f"real_gr_score_{suffix}"] = float(real_score[sign])
            row[f"circular_gr_score_{suffix}"] = float(circular_score[sign])
        score_rows.append(row)
    score_frame = pd.DataFrame(score_rows)
    numeric = candidate_rows.drop(columns=["id", "well_id"]).to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"Non-finite candidate path rows for well={well}")
    if len(score_frame):
        score_numeric = score_frame.select_dtypes(include=[np.number]).to_numpy(
            np.float64
        )
        if not np.isfinite(score_numeric).all():
            raise ValueError(f"Non-finite block score rows for well={well}")
    manifest = {
        "well_id": well,
        "status": "ok" if len(score_frame) else "skipped_no_complete_block",
        "prefix_rows": int(len(known_indices)),
        "suffix_rows": int(len(eval_indices)),
        "complete_blocks": int(len(score_frame)),
        "last_known_row_index": last_known_index,
        "last_known_tvt": float(last_known["TVT_input"]),
        "last_known_md": float(last_known["MD"]),
        "initial_rate": float(initial_rate),
        "initial_rate_valid_steps": int(valid_rate_steps),
        "gr_sigma": float(sigma),
        "horizontal_raw_sha256": sha256_file(horizontal_path),
        "typewell_raw_sha256": sha256_file(typewell_path),
    }
    return candidate_rows, score_frame, manifest


# %% [markdown]
# ## 5. Truth-free candidate generation and SHA freeze
#
# Candidate paths, real/circular GR scores, input identity, schema, and the
# frozen contract are written and read back before the ledger permits truth.

# %%
@dataclass(frozen=True)
class FrozenStage0:
    candidate_path: Path
    block_score_path: Path
    input_manifest_path: Path
    freeze_manifest_path: Path
    candidate_report: dict[str, Any]
    block_score_report: dict[str, Any]
    input_manifest_report: dict[str, Any]
    contract_sha256: str


def generate_and_freeze_stage0(
    train_dir: Path,
    artifacts_dir: Path,
    config: dict[str, Any],
    contract: dict[str, Any],
    ledger: TruthAccessLedger,
    maximum_wells: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, FrozenStage0]:
    wells = discover_wells(train_dir, maximum_wells=maximum_wells)
    if not wells:
        raise RuntimeError("No paired train wells were found")
    candidate_parts: list[pd.DataFrame] = []
    score_parts: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    for index, well in enumerate(wells, start=1):
        candidate, scores, manifest = build_prefreeze_rows_for_well(
            well,
            train_dir,
            config,
            ledger,
        )
        manifest_rows.append(manifest)
        if len(candidate) and len(scores):
            candidate_parts.append(candidate)
            score_parts.append(scores)
        if index % 50 == 0 or index == len(wells):
            print(
                f"Stage 0 truth-free generation: {index}/{len(wells)} wells",
                flush=True,
            )
    if not candidate_parts or not score_parts:
        raise RuntimeError("Stage 0 generated no complete 512-row blocks")
    candidate_frame = (
        pd.concat(candidate_parts, ignore_index=True)
        .sort_values(["well_id", "row_index"])
        .reset_index(drop=True)
    )
    block_scores = (
        pd.concat(score_parts, ignore_index=True)
        .sort_values(["well_id", "block_index"])
        .reset_index(drop=True)
    )
    input_manifest = (
        pd.DataFrame(manifest_rows)
        .sort_values("well_id")
        .reset_index(drop=True)
    )
    if candidate_frame["id"].duplicated().any():
        raise ValueError("Candidate path IDs must be unique")
    if block_scores.duplicated(["well_id", "block_index"]).any():
        raise ValueError("Block score keys must be unique")

    candidate_path = artifacts_dir / f"{OUTPUT_PREFIX}_candidate_paths.csv.gz"
    score_path = artifacts_dir / f"{OUTPUT_PREFIX}_block_gr_scores.csv.gz"
    input_path = artifacts_dir / f"{OUTPUT_PREFIX}_input_manifest.csv"
    freeze_path = artifacts_dir / f"{OUTPUT_PREFIX}_freeze_manifest.json"
    write_deterministic_csv_gzip(candidate_frame, candidate_path)
    write_deterministic_csv_gzip(block_scores, score_path)
    input_manifest.to_csv(input_path, index=False)
    candidate_report = artifact_sha_report(candidate_path)
    candidate_report["schema_sha256"] = dataframe_schema_sha256(candidate_frame)
    block_score_report = artifact_sha_report(score_path)
    block_score_report["schema_sha256"] = dataframe_schema_sha256(block_scores)
    input_report = artifact_sha_report(input_path)
    input_report["schema_sha256"] = dataframe_schema_sha256(input_manifest)
    freeze_payload = {
        "experiment": EXPERIMENT_NAME,
        "phase": "stage0_truth_freeze",
        "created_at": datetime.now(UTC).isoformat(),
        "contract_sha256": contract["contract_sha256"],
        "candidate_paths": candidate_report,
        "block_gr_scores": block_score_report,
        "input_manifest": input_report,
        "candidate_rows": int(len(candidate_frame)),
        "blocks": int(len(block_scores)),
        "wells": int(block_scores["well_id"].nunique()),
        "truth_rows_before_freeze": int(ledger.truth_rows_before_freeze),
        "hidden_role_rows_before_freeze": int(
            ledger.hidden_role_rows_before_freeze
        ),
    }
    write_json(freeze_path, freeze_payload)
    frozen = FrozenStage0(
        candidate_path=candidate_path,
        block_score_path=score_path,
        input_manifest_path=input_path,
        freeze_manifest_path=freeze_path,
        candidate_report=candidate_report,
        block_score_report=block_score_report,
        input_manifest_report=input_report,
        contract_sha256=str(contract["contract_sha256"]),
    )
    verify_frozen_stage0(frozen)
    ledger.freeze()
    return candidate_frame, block_scores, input_manifest, frozen


def verify_frozen_stage0(frozen: FrozenStage0) -> None:
    current_candidate = artifact_sha_report(frozen.candidate_path)
    current_scores = artifact_sha_report(frozen.block_score_path)
    current_inputs = artifact_sha_report(frozen.input_manifest_path)
    if (
        current_candidate["content_sha256"]
        != frozen.candidate_report["content_sha256"]
    ):
        raise ValueError("Candidate path content changed after freeze")
    if (
        current_scores["content_sha256"]
        != frozen.block_score_report["content_sha256"]
    ):
        raise ValueError("Block GR score content changed after freeze")
    if (
        current_inputs["content_sha256"]
        != frozen.input_manifest_report["content_sha256"]
    ):
        raise ValueError("Input manifest content changed after freeze")
    payload = json.loads(frozen.freeze_manifest_path.read_text())
    if payload["contract_sha256"] != frozen.contract_sha256:
        raise ValueError("Frozen contract SHA mismatch")


def read_frozen_stage0(
    frozen: FrozenStage0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    verify_frozen_stage0(frozen)
    candidate_paths = pd.read_csv(
        frozen.candidate_path,
        dtype={"id": str, "well_id": str},
    )
    block_scores = pd.read_csv(
        frozen.block_score_path,
        dtype={
            "well_id": str,
            "start_id": str,
            "stop_id": str,
            "real_rank_order": str,
            "circular_rank_order": str,
        },
    )
    if candidate_paths["id"].duplicated().any():
        raise ValueError("Frozen candidate path IDs must remain unique")
    if block_scores.duplicated(["well_id", "block_index"]).any():
        raise ValueError("Frozen block score keys must remain unique")
    if list(candidate_paths.columns) != [
        "id",
        "well_id",
        "row_index",
        "suffix_row",
        "md_since",
        "path_minus",
        "path_zero",
        "path_plus",
    ]:
        raise ValueError("Frozen candidate path schema changed after CSV readback")
    required_score_columns = {
        "well_id",
        "block_index",
        "start_suffix_row",
        "stop_suffix_row_exclusive",
        "block_rows",
        "real_top1_sign",
        "circular_top1_sign",
        "real_rank_order",
        "circular_rank_order",
    }
    if not required_score_columns.issubset(block_scores.columns):
        missing = sorted(required_score_columns - set(block_scores.columns))
        raise ValueError(f"Frozen block score schema is missing: {missing}")
    return candidate_paths, block_scores


# %% [markdown]
# ## 6. Late truth and hidden-like attachment
#
# Horizontal target and exp115 roles are loaded only after the candidate and
# score artifacts pass SHA readback.

# %%
def load_truth_late(
    wells: Sequence[str],
    train_dir: Path,
    ledger: TruthAccessLedger,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for well in sorted(set(str(value) for value in wells)):
        path = train_dir / f"{well}__horizontal_well.csv"
        horizontal = pd.read_csv(path, usecols=["TVT", "TVT_input"])
        eval_indices = np.flatnonzero(horizontal["TVT_input"].isna().to_numpy())
        truth = pd.to_numeric(
            horizontal.loc[eval_indices, "TVT"], errors="coerce"
        ).to_numpy(np.float64)
        if not np.isfinite(truth).all():
            raise ValueError(f"Non-finite suffix truth for well={well}")
        rows.append(
            pd.DataFrame(
                {
                    "id": [f"{well}_{int(row_index)}" for row_index in eval_indices],
                    "true_tvt": truth.astype(np.float32),
                }
            )
        )
    frame = pd.concat(rows, ignore_index=True)
    if frame["id"].duplicated().any():
        raise ValueError("Late truth IDs must be unique")
    ledger.record_truth_late(len(frame))
    return frame


def load_hidden_like_late(
    config: dict[str, Any],
    ledger: TruthAccessLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    hidden = get_nested(config, "data.hidden_like") or {}
    path = resolve_existing_file(
        candidates=list(hidden.get("candidates") or []),
        patterns=list(hidden.get("patterns") or []),
        filename=str(hidden["filename"]),
    )
    actual_sha = sha256_file(path)
    expected_sha = str(hidden["expected_sha256"])
    if actual_sha != expected_sha:
        raise ValueError(
            f"Hidden-like assignment SHA mismatch: expected={expected_sha} actual={actual_sha}"
        )
    role_columns = list((hidden.get("role_columns") or {}).values())
    frame = pd.read_csv(path, dtype={"well_id": str})
    missing = sorted({"well_id", *role_columns} - set(frame.columns))
    if missing:
        raise ValueError(f"Hidden-like assignment missing columns: {missing}")
    frame = frame[["well_id", *role_columns]].copy()
    if frame["well_id"].duplicated().any():
        raise ValueError("Hidden-like assignment must have one row per well")
    ledger.record_hidden_late(len(frame))
    return frame, {
        "path": str(path),
        "raw_sha256": actual_sha,
        "rows": int(len(frame)),
        "role_columns": role_columns,
    }


def parse_rank_order(value: str) -> list[int]:
    return [int(part) for part in str(value).split(",")]


def build_postfreeze_block_readout(
    candidate_paths: pd.DataFrame,
    block_scores: pd.DataFrame,
    truth: pd.DataFrame,
    hidden_like: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    merged = candidate_paths.merge(truth, on="id", how="left", validate="one_to_one")
    if merged["true_tvt"].isna().any():
        raise ValueError("Late truth join left missing candidate rows")
    role_columns = [
        str(value)
        for value in (get_nested(config, "data.hidden_like.role_columns") or {}).values()
    ]
    merged = merged.merge(
        hidden_like,
        on="well_id",
        how="left",
        validate="many_to_one",
    )
    if merged[role_columns].isna().any().any():
        raise ValueError("Hidden-like late join left missing roles")
    stage0 = get_nested(config, "validation.stage_0") or {}
    signs = [int(value) for value in stage0["candidate_signs"]]
    zero_rank = [int(value) for value in stage0["zero_first_reference_rank"]]
    block_rows: list[dict[str, Any]] = []
    by_well = {
        well: frame.sort_values("suffix_row").reset_index(drop=True)
        for well, frame in merged.groupby("well_id", sort=True)
    }
    for score in block_scores.itertuples(index=False):
        well = str(score.well_id)
        frame = by_well[well]
        start = int(score.start_suffix_row)
        stop = int(score.stop_suffix_row_exclusive)
        block = frame.iloc[start:stop]
        if len(block) != int(score.block_rows):
            raise ValueError(f"Block row-count mismatch for well={well}")
        true = block["true_tvt"].to_numpy(np.float64)
        mse: dict[int, float] = {}
        rmse: dict[int, float] = {}
        for sign in signs:
            prediction = block[f"path_{SIGN_COLUMN[sign]}"].to_numpy(np.float64)
            mse[sign] = float(np.mean((prediction - true) ** 2))
            rmse[sign] = float(np.sqrt(mse[sign]))
        oracle_rank = sorted(
            signs,
            key=lambda sign: (rmse[sign], zero_rank.index(sign)),
        )
        oracle_sign = int(oracle_rank[0])
        real_rank = parse_rank_order(score.real_rank_order)
        circular_rank = parse_rank_order(score.circular_rank_order)
        real_sign = int(score.real_top1_sign)
        circular_sign = int(score.circular_top1_sign)
        row: dict[str, Any] = {
            "well_id": well,
            "block_index": int(score.block_index),
            "start_suffix_row": start,
            "stop_suffix_row_exclusive": stop,
            "block_rows": int(len(block)),
            "block_mid_md_since": float(score.block_mid_md_since),
            "oracle_sign": oracle_sign,
            "real_top1_sign": real_sign,
            "circular_top1_sign": circular_sign,
            "real_top1_correct": bool(real_sign == oracle_sign),
            "circular_top1_correct": bool(circular_sign == oracle_sign),
            "zero_top1_correct": bool(oracle_sign == 0),
            "real_reciprocal_rank": float(1.0 / (real_rank.index(oracle_sign) + 1)),
            "circular_reciprocal_rank": float(
                1.0 / (circular_rank.index(oracle_sign) + 1)
            ),
            "zero_first_reciprocal_rank": float(
                1.0 / (zero_rank.index(oracle_sign) + 1)
            ),
            "selected_mse": float(mse[real_sign]),
            "circular_selected_mse": float(mse[circular_sign]),
            "zero_mse": float(mse[0]),
            "selected_block_rmse": float(rmse[real_sign]),
            "zero_block_rmse": float(rmse[0]),
            "selected_block_rmse_gain_vs_zero_ft": float(
                rmse[0] - rmse[real_sign]
            ),
        }
        for sign in signs:
            row[f"path_rmse_{SIGN_COLUMN[sign]}"] = rmse[sign]
        for role_column in role_columns:
            row[role_column] = str(block.iloc[0][role_column])
        block_rows.append(row)
    result = pd.DataFrame(block_rows).sort_values(
        ["well_id", "block_index"]
    ).reset_index(drop=True)
    numeric = result.select_dtypes(include=[np.number]).to_numpy(np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Post-freeze block readout contains non-finite values")
    return result


# %% [markdown]
# ## 7. Stage 0 metrics and gates
#
# Overall thresholds use block-level top1/MRR. Fold direction requires positive
# top1, MRR, and real-vs-circular gains. The 1000+ and two hidden-like scopes use
# the frozen GR selection's pooled block RMSE gain over the zero path.

# %%
def assign_group_folds(
    block_readout: pd.DataFrame,
    requested_splits: int,
) -> pd.DataFrame:
    frame = block_readout.copy()
    wells = int(frame["well_id"].nunique())
    splits = min(int(requested_splits), wells)
    if splits < 2:
        frame["fold"] = 0
        return frame
    splitter = GroupKFold(n_splits=splits)
    fold_values = np.full(len(frame), -1, dtype=np.int16)
    dummy = np.zeros((len(frame), 1), dtype=np.float32)
    groups = frame["well_id"].astype(str).to_numpy()
    for fold, (_, valid_index) in enumerate(splitter.split(dummy, groups=groups)):
        fold_values[valid_index] = int(fold)
    if np.any(fold_values < 0):
        raise RuntimeError("GroupKFold left unassigned blocks")
    frame["fold"] = fold_values
    return frame


def ranking_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "blocks": 0,
            "rows_with_overlap": 0,
            "wells": 0,
            "top1_accuracy": None,
            "zero_first_top1_accuracy": None,
            "top1_gain_vs_zero_first": None,
            "circular_top1_accuracy": None,
            "real_minus_circular_top1": None,
            "mrr": None,
            "zero_first_mrr": None,
            "mrr_gain_vs_zero_first": None,
            "circular_mrr": None,
            "selected_path_rmse_ft": None,
            "zero_path_rmse_ft": None,
            "selected_path_rmse_gain_vs_zero_ft": None,
        }
    weights = frame["block_rows"].to_numpy(np.float64)
    selected_rmse = float(
        np.sqrt(np.average(frame["selected_mse"].to_numpy(np.float64), weights=weights))
    )
    zero_rmse = float(
        np.sqrt(np.average(frame["zero_mse"].to_numpy(np.float64), weights=weights))
    )
    top1 = float(frame["real_top1_correct"].mean())
    zero_top1 = float(frame["zero_top1_correct"].mean())
    circular_top1 = float(frame["circular_top1_correct"].mean())
    mrr = float(frame["real_reciprocal_rank"].mean())
    zero_mrr = float(frame["zero_first_reciprocal_rank"].mean())
    circular_mrr = float(frame["circular_reciprocal_rank"].mean())
    return {
        "blocks": int(len(frame)),
        "rows_with_overlap": int(frame["block_rows"].sum()),
        "wells": int(frame["well_id"].nunique()),
        "top1_accuracy": top1,
        "zero_first_top1_accuracy": zero_top1,
        "top1_gain_vs_zero_first": top1 - zero_top1,
        "circular_top1_accuracy": circular_top1,
        "real_minus_circular_top1": top1 - circular_top1,
        "mrr": mrr,
        "zero_first_mrr": zero_mrr,
        "mrr_gain_vs_zero_first": mrr - zero_mrr,
        "circular_mrr": circular_mrr,
        "selected_path_rmse_ft": selected_rmse,
        "zero_path_rmse_ft": zero_rmse,
        "selected_path_rmse_gain_vs_zero_ft": zero_rmse - selected_rmse,
    }


def build_scope_metrics(
    block_readout: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    hidden = get_nested(config, "data.hidden_like") or {}
    roles = hidden.get("role_columns") or {}
    scopes: list[tuple[str, np.ndarray]] = [
        ("overall", np.ones(len(block_readout), dtype=bool)),
        (
            "1000_plus",
            block_readout["block_mid_md_since"].to_numpy(np.float64) >= 1000.0,
        ),
    ]
    for scope_name, role_column in roles.items():
        scopes.append(
            (
                str(scope_name),
                block_readout[str(role_column)].astype(str).eq("valid").to_numpy(),
            )
        )
    rows: list[dict[str, Any]] = []
    for scope, mask in scopes:
        rows.append({"scope": scope, **ranking_metrics(block_readout.loc[mask])})
    return pd.DataFrame(rows)


def build_fold_metrics(block_readout: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold, frame in block_readout.groupby("fold", sort=True):
        metric = ranking_metrics(frame)
        passed = bool(
            metric["top1_gain_vs_zero_first"] is not None
            and metric["top1_gain_vs_zero_first"] > 0.0
            and metric["mrr_gain_vs_zero_first"] > 0.0
            and metric["real_minus_circular_top1"] > 0.0
        )
        rows.append({"fold": int(fold), **metric, "direction_pass": passed})
    return pd.DataFrame(rows)


def evaluate_stage0_gates(
    candidate_paths: pd.DataFrame,
    block_scores: pd.DataFrame,
    block_readout: pd.DataFrame,
    scope_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    ledger: TruthAccessLedger,
    frozen: FrozenStage0,
    config: dict[str, Any],
    debug: bool,
) -> dict[str, Any]:
    gates = get_nested(config, "validation.stage_0.all_required") or {}
    overall = scope_metrics.loc[scope_metrics["scope"].eq("overall")].iloc[0]
    scope_lookup = scope_metrics.set_index("scope")
    direction_scopes = [
        "1000_plus",
        "verification_like_spatial",
        "verification_like_typewell_purged",
    ]
    technical = {
        "truth_rows_before_freeze_zero": ledger.truth_rows_before_freeze == 0,
        "hidden_role_rows_before_freeze_zero": (
            ledger.hidden_role_rows_before_freeze == 0
        ),
        "candidate_sha_readback": (
            artifact_sha_report(frozen.candidate_path)["content_sha256"]
            == frozen.candidate_report["content_sha256"]
        ),
        "block_score_sha_readback": (
            artifact_sha_report(frozen.block_score_path)["content_sha256"]
            == frozen.block_score_report["content_sha256"]
        ),
        "candidate_ids_unique": not candidate_paths["id"].duplicated().any(),
        "block_keys_unique": not block_scores.duplicated(
            ["well_id", "block_index"]
        ).any(),
        "complete_512_row_blocks_only": bool(
            block_scores["block_rows"].eq(512).all()
        ),
        "candidate_block_well_identity": (
            set(block_scores["well_id"]) <= set(candidate_paths["well_id"])
        ),
        "stage0_pf_seed_well_runs_zero": (
            int(get_nested(config, "validation.stage_0.pf_seed_well_runs")) == 0
        ),
        "stage1_disabled": not bool(get_nested(config, "execution.run_stage_1")),
    }
    scientific = {
        "overall_top1": float(overall["top1_accuracy"])
        >= float(gates["minimum_top1_accuracy"]),
        "overall_mrr_gain": float(overall["mrr_gain_vs_zero_first"])
        >= float(gates["minimum_mrr_gain_vs_zero_first_rank"]),
        "overall_real_minus_circular_top1": float(
            overall["real_minus_circular_top1"]
        )
        >= float(gates["minimum_real_minus_circular_top1"]),
        "minimum_passing_folds": int(fold_metrics["direction_pass"].sum())
        >= int(gates["minimum_passing_folds"]),
        **{
            f"{scope}_positive_direction": bool(
                scope in scope_lookup.index
                and int(scope_lookup.loc[scope, "blocks"]) > 0
                and float(
                    scope_lookup.loc[
                        scope,
                        "selected_path_rmse_gain_vs_zero_ft",
                    ]
                )
                > 0.0
            )
            for scope in direction_scopes
        },
    }
    return {
        "technical": technical,
        "scientific": scientific,
        "technical_pass": bool(all(technical.values())),
        "scientific_pass": bool(all(scientific.values())),
        "stage0_pass": bool(
            not debug and all(technical.values()) and all(scientific.values())
        ),
        "debug_never_promotes": bool(debug),
        "decision_if_run": (
            "STAGE0_PASS_REQUEST_SEPARATE_STAGE1_APPROVAL"
            if not debug and all(technical.values()) and all(scientific.values())
            else "STAGE0_FAIL_CLOSE_WITHOUT_RESCUE"
            if not debug
            else "DEBUG_ONLY_NO_DECISION"
        ),
    }


# %% [markdown]
# ## 8. Execution orchestration and generated artifacts
#
# The final cell exposes all inputs, fixed counts, execution selection, freeze
# order, metrics, and outputs. No Stage 1 PF or inference path is reachable.

# %%
def run_stage0(
    config: dict[str, Any],
    package_dir: Path,
    maximum_wells: int | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    started = time.time()
    require_authoritative_runtime()
    contract = validate_scientific_contract(config)
    train_dir = resolve_train_dir(config)
    artifacts_dir, _ = output_dirs(package_dir)
    ledger = TruthAccessLedger()
    print("Experiment:", EXPERIMENT_NAME)
    print("Route:", get_nested(config, "experiment.route"))
    print("Parent:", get_nested(config, "lineage.parent"))
    print("Implementation scope:", get_nested(config, "execution.implementation_scope"))
    print("Raw train:", train_dir)
    print("Contract SHA256:", contract["contract_sha256"])
    print("Stage 0 PF seed-well runs: 0")
    print("Stage 1 planned seed-well runs: 98,944 (disabled)")
    print("LightGBM configs / trained folds / boosters: 0 / 0 / 0")

    _, _, input_manifest, frozen = (
        generate_and_freeze_stage0(
            train_dir=train_dir,
            artifacts_dir=artifacts_dir,
            config=config,
            contract=contract,
            ledger=ledger,
            maximum_wells=maximum_wells,
        )
    )
    candidate_paths, block_scores = read_frozen_stage0(frozen)
    truth = load_truth_late(
        candidate_paths["well_id"].unique().tolist(),
        train_dir,
        ledger,
    )
    hidden_like, hidden_report = load_hidden_like_late(config, ledger)
    block_readout = build_postfreeze_block_readout(
        candidate_paths,
        block_scores,
        truth,
        hidden_like,
        config,
    )
    block_readout = assign_group_folds(
        block_readout,
        int(get_nested(config, "validation.n_folds")),
    )
    scope_metrics = build_scope_metrics(block_readout, config)
    fold_metrics = build_fold_metrics(block_readout)
    gate_report = evaluate_stage0_gates(
        candidate_paths,
        block_scores,
        block_readout,
        scope_metrics,
        fold_metrics,
        ledger,
        frozen,
        config,
        debug,
    )

    readout_path = artifacts_dir / f"{OUTPUT_PREFIX}_postfreeze_block_readout.csv.gz"
    scope_path = artifacts_dir / f"{OUTPUT_PREFIX}_scope_metrics.csv"
    fold_path = artifacts_dir / f"{OUTPUT_PREFIX}_fold_metrics.csv"
    gate_path = artifacts_dir / f"{OUTPUT_PREFIX}_gate_report.json"
    summary_path = artifacts_dir / f"{OUTPUT_PREFIX}_summary.json"
    write_deterministic_csv_gzip(block_readout, readout_path)
    scope_metrics.to_csv(scope_path, index=False)
    fold_metrics.to_csv(fold_path, index=False)
    write_json(gate_path, gate_report)
    artifact_reports = {
        "candidate_paths": frozen.candidate_report,
        "block_gr_scores": frozen.block_score_report,
        "input_manifest": frozen.input_manifest_report,
        "freeze_manifest": artifact_sha_report(frozen.freeze_manifest_path),
        "postfreeze_block_readout": artifact_sha_report(readout_path),
        "scope_metrics": artifact_sha_report(scope_path),
        "fold_metrics": artifact_sha_report(fold_path),
        "gate_report": artifact_sha_report(gate_path),
    }
    overall = (
        scope_metrics.loc[scope_metrics["scope"].eq("overall")]
        .iloc[0]
        .to_dict()
    )
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
        "wells_discovered": int(len(input_manifest)),
        "wells_scored": int(block_scores["well_id"].nunique()),
        "candidate_rows": int(len(candidate_paths)),
        "blocks": int(len(block_readout)),
        "truth_access": {
            "truth_rows_before_freeze": ledger.truth_rows_before_freeze,
            "hidden_role_rows_before_freeze": ledger.hidden_role_rows_before_freeze,
            "truth_rows_after_freeze": ledger.truth_rows_after_freeze,
            "hidden_role_rows_after_freeze": ledger.hidden_role_rows_after_freeze,
        },
        "hidden_like": hidden_report,
        "overall": overall,
        "passing_folds": int(fold_metrics["direction_pass"].sum()),
        "gates": gate_report,
        "execution_counts": {
            "scientific_variants": 0,
            "fixed_signed_paths": 3,
            "pf_seed_well_runs": 0,
            "pf_control_replays": 0,
            "lightgbm_configs": 0,
            "trained_folds": 0,
            "boosters": 0,
        },
        "artifacts": artifact_reports,
    }
    write_json(summary_path, summary)
    printable_artifact_reports = {
        **artifact_reports,
        "summary": artifact_sha_report(summary_path),
    }

    metrics_path = (
        KAGGLE_WORKING_ROOT / "metrics.json"
        if is_kaggle_runtime()
        else package_dir / "metrics.json"
    )
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": summary["status"],
        "updated_at": datetime.now(UTC).date().isoformat(),
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "metric": "rmse",
        "stage0": {
            "overall": overall,
            "passing_folds": summary["passing_folds"],
            "gates": gate_report,
            "summary_path": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
        },
        "notes": (
            "Stage 0 only; Stage 1 PF, inference, and submission remain disabled."
        ),
    }
    write_json(metrics_path, metrics)
    print(scope_metrics.to_string(index=False))
    print(fold_metrics.to_string(index=False))
    print(json.dumps(gate_report, indent=2))
    print("Generated artifacts:")
    for name, report in printable_artifact_reports.items():
        print(f"  {name}: {report['path']} | content_sha={report['content_sha256']}")
    return summary


# %%
if __name__ == "__main__":
    PACKAGE_DIR = resolve_package_dir()
    CONFIG = load_config(PACKAGE_DIR)
    DEBUG = os.environ.get("EXPERIMENT_DEBUG", "0") == "1"
    MAX_WELLS_ENV = os.environ.get("EXPERIMENT_MAX_WELLS")
    MAX_WELLS = int(MAX_WELLS_ENV) if MAX_WELLS_ENV else None

    # A run requires separate config approval. This protects the implementation
    # turn from accidentally executing the Kaggle experiment.
    if not bool(get_nested(CONFIG, "execution.run_stage_0")):
        raise RuntimeError(
            "Stage 0 is implemented but execution.run_stage_0 is false. "
            "Obtain separate Kaggle push/run approval before enabling it."
        )
    if not bool(get_nested(CONFIG, "execution.kaggle_push_approved")):
        raise RuntimeError(
            "Stage 0 execution requires execution.kaggle_push_approved=true."
        )
    STAGE0_SUMMARY = run_stage0(
        CONFIG,
        package_dir=PACKAGE_DIR,
        maximum_wells=MAX_WELLS,
        debug=DEBUG,
    )
