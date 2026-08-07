# %% [markdown]
# # exp364 signed-curvature exact HMM — Stage 0 train-side preflight
#
# This compact self-contained notebook implements only the design-frozen
# Stage 0. It builds three deterministic signed-curvature paths with the
# unchanged exp209 prefix-rate and GR-emission contract, freezes paths, scores,
# input identity, and a 16-well exact-state resource projection before reading
# suffix truth, then evaluates the frozen ranking. It does not decode the
# curvature HMM, rerun exp209, train a model, or create a submission.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Notebook-safe runtime, configuration, path, and SHA helpers
# 3. Frozen scientific and execution contract
# 4. Truth-free exp209 input and signed-path helpers
# 5. Sixteen-well exact-state resource projection
# 6. Candidate generation and pre-truth SHA freeze
# 7. Late truth and hidden-like attachment
# 8. Stage 0 metrics and promotion gates
# 9. Execution orchestration and generated artifacts
# 10. Setup and fail-closed execution selection

# %% [markdown]
# ## 1. Imports

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

EXPERIMENT_NAME = "exp364_signed_curvature_exact_hmm"
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
# ## 2. Notebook-safe runtime, configuration, path, and SHA helpers

# %%
def get_nested(mapping: dict[str, Any], dotted_key: str) -> Any:
    value: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
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
    candidates = [cwd, cwd / "experiments" / EXPERIMENT_NAME, KAGGLE_WORKING_ROOT]
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
            config = yaml.safe_load(path.read_text()) or {}
        except (OSError, yaml.YAMLError):
            continue
        if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
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
        "exp364 Stage 0 must run in Kaggle. Set EXPERIMENT_ALLOW_LOCAL=1 only "
        "for an explicitly approved local smoke run."
    )


def resolve_train_dir(config: dict[str, Any]) -> Path:
    configured = Path(str(get_nested(config, "data.train_dir") or "data/raw/train"))
    candidates = [
        configured,
        Path.cwd() / configured,
        Path.cwd() / "data" / "raw" / "train",
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
# ## 3. Frozen scientific and execution contract

# %%
def stage0_contract(config: dict[str, Any]) -> dict[str, Any]:
    stage0 = get_nested(config, "validation.stage_0") or {}
    signed = get_nested(config, "model.signed_curvature") or {}
    fixed = get_nested(config, "model.fixed_from_exp209") or {}
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
        "resource_projection_wells": stage0.get("resource_projection_wells"),
        "resource_projection": stage0.get("resource_projection"),
        "curvature_states": signed.get("states"),
        "rate_drift_per_row": signed.get("rate_drift_per_row"),
        "initial_probability": signed.get("initial_probability"),
        "transition_matrix": signed.get("transition_matrix"),
        "position_grid_step_ft": fixed.get("position_grid_step_ft"),
        "n_rates": fixed.get("n_rates"),
        "rate_span": fixed.get("rate_span"),
        "sig_r": fixed.get("sig_r"),
        "sig_p": fixed.get("sig_p"),
        "emission": fixed.get("emission"),
        "emission_squared_z_clip": fixed.get("emission_squared_z_clip"),
        "sigma_mode": fixed.get("sigma_mode"),
        "gr_sigma_min": fixed.get("gr_sigma_min"),
        "gr_sigma_max": fixed.get("gr_sigma_max"),
        "gr_sigma_default": fixed.get("gr_sigma_default"),
        "initial_rate_window_rows": fixed.get("initial_rate_window_rows"),
        "initial_rate_min_valid_steps": fixed.get(
            "initial_rate_min_valid_steps"
        ),
        "initial_rate_fallback": fixed.get("initial_rate_fallback"),
        "band_pad_ft": fixed.get("band_pad_ft"),
        "typewell_outer_pad_ft": fixed.get("typewell_outer_pad_ft"),
        "momentum": fixed.get("momentum"),
        "rate_center": fixed.get("rate_center"),
        "output": fixed.get("output"),
        "hmm_well_runs": stage0.get("hmm_well_runs"),
    }
    contract["contract_sha256"] = mapping_sha256(contract)
    return contract


def validate_scientific_contract(
    config: dict[str, Any],
    *,
    require_run_approval: bool = False,
) -> dict[str, Any]:
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
        "resource_projection_wells": 16,
        "curvature_states": [-1, 0, 1],
        "rate_drift_per_row": 0.000009765625,
        "initial_probability": [0.25, 0.50, 0.25],
        "position_grid_step_ft": 0.35,
        "n_rates": 41,
        "rate_span": 0.10,
        "sig_r": 0.002,
        "sig_p": 0.02,
        "emission": "gauss",
        "emission_squared_z_clip": 600.0,
        "sigma_mode": "std",
        "gr_sigma_min": 10.0,
        "gr_sigma_max": 60.0,
        "gr_sigma_default": 30.0,
        "initial_rate_window_rows": 30,
        "initial_rate_min_valid_steps": 3,
        "initial_rate_fallback": 0.0,
        "band_pad_ft": 100.0,
        "typewell_outer_pad_ft": 40.0,
        "momentum": 0.998,
        "rate_center": "zero",
        "output": "posterior_mean",
        "hmm_well_runs": 0,
    }
    mismatches = {
        key: {"expected": expected, "actual": contract.get(key)}
        for key, expected in required_equal.items()
        if contract.get(key) != expected
    }
    transition = np.asarray(contract["transition_matrix"], dtype=np.float64)
    expected_transition = np.asarray(
        [
            [511.0 / 512.0, 1.0 / 512.0, 0.0],
            [1.0 / 2048.0, 1023.0 / 1024.0, 1.0 / 2048.0],
            [0.0, 1.0 / 512.0, 511.0 / 512.0],
        ],
        dtype=np.float64,
    )
    if transition.shape != (3, 3) or not np.array_equal(
        transition, expected_transition
    ):
        mismatches["transition_matrix"] = {
            "expected": expected_transition.tolist(),
            "actual": transition.tolist(),
        }
    if mismatches:
        raise ValueError(f"Frozen exp364 Stage 0 contract mismatch: {mismatches}")

    forbidden = set(get_nested(config, "model.forbidden") or [])
    expected_forbidden = {
        "prefix_or_geometry_rate_regressor",
        "curvature_magnitude_or_transition_grid",
        "continuous_acceleration",
        "emission_or_sigma_change",
        "adaptive_rate_noise",
        "blend_or_selector",
        "parent_control_rerun",
    }
    if forbidden != expected_forbidden:
        raise ValueError("exp364 forbidden-operation contract changed")
    counts = get_nested(config, "execution.stage_0_counts") or {}
    expected_counts = {
        "diagnostic_variants": 1,
        "fixed_signed_paths": 3,
        "reporting_folds": 5,
        "resource_projection_wells": 16,
        "exact_hmm_well_runs": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "parent_control_reruns": 0,
    }
    if counts != expected_counts:
        raise ValueError("exp364 Stage 0 execution counts changed")
    if require_run_approval and not (
        bool(get_nested(config, "execution.kaggle_push_approved"))
        and bool(get_nested(config, "execution.run_stage_0"))
    ):
        raise PermissionError("exp364 Kaggle Stage 0 run is not approved")
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
            raise ValueError(
                f"{label}: forbidden pre-freeze columns requested: {sorted(overlap)}"
            )

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
# ## 4. Truth-free exp209 input and signed-path helpers

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


def prepare_typewell(
    typewell: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
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


def exp209_gr_sigma(
    horizontal: pd.DataFrame,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    sigma_min: float,
    sigma_max: float,
    sigma_default: float,
) -> float:
    known = horizontal.loc[horizontal["TVT_input"].notna()]
    if len(known) == 0:
        return float(sigma_max)
    observed = (
        pd.to_numeric(known["GR"], errors="coerce")
        .fillna(0.0)
        .to_numpy(np.float64)
    )
    known_tvt = pd.to_numeric(
        known["TVT_input"], errors="coerce"
    ).to_numpy(np.float64)
    residual = observed - np.interp(known_tvt, typewell_tvt, typewell_gr)
    sigma = float(np.nanstd(residual))
    if not np.isfinite(sigma):
        return float(sigma_default)
    return float(np.clip(sigma, sigma_min, sigma_max))


def exp209_initial_rate(
    known: pd.DataFrame,
    window_rows: int,
    minimum_valid_steps: int,
    fallback: float,
) -> tuple[float, int]:
    tail = known.tail(int(window_rows))
    tvt = pd.to_numeric(tail["TVT_input"], errors="coerce").to_numpy(np.float64)
    z = pd.to_numeric(tail["Z"], errors="coerce").to_numpy(np.float64)
    md = pd.to_numeric(tail["MD"], errors="coerce").to_numpy(np.float64)
    dtvt = np.diff(tvt)
    dz = np.diff(z)
    dmd = np.diff(md)
    valid = np.isfinite(dtvt) & np.isfinite(dz) & np.isfinite(dmd) & (dmd > 0)
    count = int(valid.sum())
    if count < int(minimum_valid_steps):
        return float(fallback), count
    return float(np.median((dtvt[valid] + dz[valid]) / dmd[valid])), count


def exp209_position_grid_bounds(
    last_known_tvt: float,
    typewell_tvt: np.ndarray,
    band_pad_ft: float,
    typewell_outer_pad_ft: float,
    step_ft: float,
) -> tuple[float, float, int]:
    grid_min = max(
        float(typewell_tvt.min()) - float(typewell_outer_pad_ft),
        float(last_known_tvt) - float(band_pad_ft),
    )
    grid_max = min(
        float(typewell_tvt.max()) + float(typewell_outer_pad_ft),
        float(last_known_tvt) + float(band_pad_ft),
    )
    grid = np.arange(grid_min, grid_max + float(step_ft), float(step_ft))
    if len(grid) < 2:
        raise ValueError("exp209 position grid contains fewer than two cells")
    return float(grid[0]), float(grid[-1]), int(len(grid))


def generate_fixed_signed_paths(
    eval_md: np.ndarray,
    eval_z: np.ndarray,
    last_known_md: float,
    last_known_position: float,
    initial_rate: float,
    signs: Sequence[int],
    momentum: float,
    rate_drift_per_row: float,
    grid_min: float,
    grid_max: float,
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
            tvt = float(np.clip(position - float(z[row_index]), grid_min, grid_max))
            position = tvt + float(z[row_index])
            path[row_index] = tvt
            previous_md = float(md[row_index])
        paths[int(sign)] = path
    return paths


def gaussian_gr_score(
    observed_gr: np.ndarray,
    path_tvt: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    sigma: float,
    squared_z_clip: float,
) -> float:
    expected_gr = np.interp(path_tvt, typewell_tvt, typewell_gr)
    residual = (np.asarray(observed_gr, np.float64) - expected_gr) / float(sigma)
    return float(np.mean(-0.5 * np.minimum(residual * residual, squared_z_clip)))


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
        return [np.roll(copied[0], int(single_block_shift_rows) % len(copied[0]))]
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
        well, train_dir, ledger
    )
    known_mask = horizontal["TVT_input"].notna().to_numpy()
    eval_mask = ~known_mask
    known_indices = np.flatnonzero(known_mask)
    eval_indices = np.flatnonzero(eval_mask)
    base_manifest = {
        "well_id": well,
        "prefix_rows": int(len(known_indices)),
        "suffix_rows": int(len(eval_indices)),
        "horizontal_raw_sha256": sha256_file(horizontal_path),
        "typewell_raw_sha256": sha256_file(typewell_path),
    }
    if len(known_indices) < 4:
        return pd.DataFrame(), pd.DataFrame(), {
            **base_manifest,
            "status": "skipped_short_prefix",
        }
    if len(eval_indices) == 0:
        return pd.DataFrame(), pd.DataFrame(), {
            **base_manifest,
            "status": "skipped_no_suffix",
        }
    if int(known_indices[-1]) >= int(eval_indices[0]):
        raise ValueError(f"well={well} does not have one contiguous visible prefix")

    stage0 = get_nested(config, "validation.stage_0") or {}
    signed = get_nested(config, "model.signed_curvature") or {}
    fixed = get_nested(config, "model.fixed_from_exp209") or {}
    signs = [int(value) for value in stage0["candidate_signs"]]
    typewell_tvt, typewell_gr = prepare_typewell(typewell)
    raw_gr = pd.to_numeric(horizontal["GR"], errors="coerce")
    observed_gr = (
        raw_gr.interpolate(limit_direction="both")
        .fillna(float(np.mean(typewell_gr)))
        .to_numpy(np.float64)
    )
    sigma = exp209_gr_sigma(
        horizontal,
        typewell_tvt,
        typewell_gr,
        float(fixed["gr_sigma_min"]),
        float(fixed["gr_sigma_max"]),
        float(fixed["gr_sigma_default"]),
    )
    known = horizontal.loc[known_mask]
    initial_rate, valid_rate_steps = exp209_initial_rate(
        known,
        int(fixed["initial_rate_window_rows"]),
        int(fixed["initial_rate_min_valid_steps"]),
        float(fixed["initial_rate_fallback"]),
    )
    last_known_index = int(known_indices[-1])
    last_known = horizontal.iloc[last_known_index]
    grid_min, grid_max, position_grid_count = exp209_position_grid_bounds(
        float(last_known["TVT_input"]),
        typewell_tvt,
        float(fixed["band_pad_ft"]),
        float(fixed["typewell_outer_pad_ft"]),
        float(fixed["position_grid_step_ft"]),
    )
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
        eval_md,
        eval_z,
        float(last_known["MD"]),
        float(last_known["TVT_input"]) + float(last_known["Z"]),
        initial_rate,
        signs,
        float(fixed["momentum"]),
        float(signed["rate_drift_per_row"]),
        grid_min,
        grid_max,
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
                typewell_tvt,
                typewell_gr,
                sigma,
                float(fixed["emission_squared_z_clip"]),
            )
            for sign in signs
        }
        circular_score = {
            sign: gaussian_gr_score(
                circular,
                paths[sign][start:stop],
                typewell_tvt,
                typewell_gr,
                sigma,
                float(fixed["emission_squared_z_clip"]),
            )
            for sign in signs
        }
        real_rank = ranked_signs(real_score, stage0["stable_score_tie_break"])
        circular_rank = ranked_signs(
            circular_score, stage0["stable_score_tie_break"]
        )
        row: dict[str, Any] = {
            "well_id": well,
            "block_index": int(block_index),
            "start_suffix_row": int(start),
            "stop_suffix_row_exclusive": int(stop),
            "start_row_index": int(eval_indices[start]),
            "stop_row_index_inclusive": int(eval_indices[stop - 1]),
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
    if not np.isfinite(
        candidate_rows.drop(columns=["id", "well_id"]).to_numpy(np.float64)
    ).all():
        raise ValueError(f"Non-finite candidate path rows for well={well}")
    manifest = {
        **base_manifest,
        "status": "ok" if len(score_frame) else "skipped_no_complete_block",
        "complete_blocks": int(len(score_frame)),
        "last_known_row_index": last_known_index,
        "last_known_tvt": float(last_known["TVT_input"]),
        "last_known_md": float(last_known["MD"]),
        "initial_rate": float(initial_rate),
        "initial_rate_valid_steps": int(valid_rate_steps),
        "gr_sigma": float(sigma),
        "position_grid_min": grid_min,
        "position_grid_max": grid_max,
        "position_grid_count": position_grid_count,
        "rate_grid_count": int(fixed["n_rates"]),
        "parent_state_cell_rows": int(
            len(eval_indices) * position_grid_count * int(fixed["n_rates"])
        ),
    }
    return candidate_rows, score_frame, manifest


# %% [markdown]
# ## 5. Sixteen-well exact-state resource projection
#
# No scientific HMM is decoded here. The projection selects 16 deterministic
# state-workload quantiles including both extrema. Runtime scales the selected
# exp209 v5 HMM wall time by the exact threefold curvature state count. Peak RSS
# is estimated from the actual exp209 tensor shapes with an explicit safety
# factor. Both values are hard Stage 0 gates.

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
        ["parent_state_cell_rows", "well_id"], kind="mergesort", inplace=True
    )
    eligible.reset_index(drop=True, inplace=True)
    if len(eligible) < int(count):
        raise ValueError(
            f"resource projection requires {count} eligible wells, got {len(eligible)}"
        )
    positions = np.linspace(0, len(eligible) - 1, int(count), dtype=np.int64)
    if len(np.unique(positions)) != int(count):
        raise RuntimeError("resource projection quantile selection is not unique")
    selected = eligible.iloc[positions].copy().reset_index(drop=True)
    selected["selection_rank"] = positions.astype(np.int32)
    selected["selection_order"] = np.arange(len(selected), dtype=np.int16)
    return selected


def build_resource_projection(
    input_manifest: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    stage0 = get_nested(config, "validation.stage_0") or {}
    spec = stage0["resource_projection"]
    selected = select_resource_projection_wells(
        input_manifest,
        int(stage0["resource_projection_wells"]),
    )
    curvature_states = int(spec["curvature_state_count"])
    alpha_bytes = int(spec["alpha_dtype_bytes"])
    posterior_bytes = int(spec["posterior_dtype_bytes"])
    emission_bytes = int(spec["emission_dtype_bytes"])
    workspace_planes = int(spec["workspace_state_planes"])
    fixed_overhead = float(spec["fixed_process_overhead_gb"]) * 1e9
    safety = float(spec["peak_rss_safety_factor"])

    selected["candidate_state_cell_rows"] = (
        selected["parent_state_cell_rows"].astype(np.int64) * curvature_states
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
        * emission_bytes
    )
    selected["workspace_bytes"] = (
        selected["position_grid_count"].astype(np.int64)
        * selected["rate_grid_count"].astype(np.int64)
        * curvature_states
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
    projected_runtime = parent_runtime * state_multiplier
    summary = {
        "method": str(spec["method"]),
        "selection": str(spec["selection"]),
        "selected_wells": int(len(selected)),
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
        "projected_runtime_seconds": projected_runtime,
        "projected_peak_rss_gb": float(selected["projected_peak_rss_gb"].max()),
        "curvature_state_count": curvature_states,
        "scientific_hmm_well_runs": 0,
        "resource_projection_wells": int(len(selected)),
    }
    return selected, summary


# %% [markdown]
# ## 6. Candidate generation and pre-truth SHA freeze

# %%
@dataclass(frozen=True)
class FrozenStage0:
    candidate_path: Path
    block_score_path: Path
    input_manifest_path: Path
    resource_projection_path: Path
    freeze_manifest_path: Path
    candidate_report: dict[str, Any]
    block_score_report: dict[str, Any]
    input_manifest_report: dict[str, Any]
    resource_projection_report: dict[str, Any]
    resource_summary: dict[str, Any]
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
            well, train_dir, config, ledger
        )
        manifest_rows.append(manifest)
        if len(candidate) and len(scores):
            candidate_parts.append(candidate)
            score_parts.append(scores)
        if index % 50 == 0 or index == len(wells):
            print(f"Stage 0 truth-free generation: {index}/{len(wells)} wells")
    if not candidate_parts or not score_parts:
        raise RuntimeError("Stage 0 generated no complete 512-row blocks")
    candidate_frame = (
        pd.concat(candidate_parts, ignore_index=True)
        .sort_values(["well_id", "row_index"], kind="mergesort")
        .reset_index(drop=True)
    )
    block_scores = (
        pd.concat(score_parts, ignore_index=True)
        .sort_values(["well_id", "block_index"], kind="mergesort")
        .reset_index(drop=True)
    )
    input_manifest = (
        pd.DataFrame(manifest_rows)
        .sort_values("well_id", kind="mergesort")
        .reset_index(drop=True)
    )
    if candidate_frame["id"].duplicated().any():
        raise ValueError("Candidate path IDs must be unique")
    if block_scores.duplicated(["well_id", "block_index"]).any():
        raise ValueError("Block score keys must be unique")
    resource_projection, resource_summary = build_resource_projection(
        input_manifest, config
    )

    candidate_path = artifacts_dir / f"{OUTPUT_PREFIX}_candidate_paths.csv.gz"
    score_path = artifacts_dir / f"{OUTPUT_PREFIX}_block_gr_scores.csv.gz"
    input_path = artifacts_dir / f"{OUTPUT_PREFIX}_input_manifest.csv"
    resource_path = artifacts_dir / f"{OUTPUT_PREFIX}_resource_projection.csv"
    freeze_path = artifacts_dir / f"{OUTPUT_PREFIX}_freeze_manifest.json"
    write_deterministic_csv_gzip(candidate_frame, candidate_path)
    write_deterministic_csv_gzip(block_scores, score_path)
    input_manifest.to_csv(input_path, index=False)
    resource_projection.to_csv(resource_path, index=False)
    reports = {
        "candidate": artifact_sha_report(candidate_path),
        "scores": artifact_sha_report(score_path),
        "inputs": artifact_sha_report(input_path),
        "resource": artifact_sha_report(resource_path),
    }
    reports["candidate"]["schema_sha256"] = dataframe_schema_sha256(candidate_frame)
    reports["scores"]["schema_sha256"] = dataframe_schema_sha256(block_scores)
    reports["inputs"]["schema_sha256"] = dataframe_schema_sha256(input_manifest)
    reports["resource"]["schema_sha256"] = dataframe_schema_sha256(
        resource_projection
    )
    freeze_payload = {
        "experiment": EXPERIMENT_NAME,
        "phase": "stage0_truth_freeze",
        "created_at": datetime.now(UTC).isoformat(),
        "contract_sha256": contract["contract_sha256"],
        "candidate_paths": reports["candidate"],
        "block_gr_scores": reports["scores"],
        "input_manifest": reports["inputs"],
        "resource_projection": reports["resource"],
        "resource_summary": resource_summary,
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
        resource_projection_path=resource_path,
        freeze_manifest_path=freeze_path,
        candidate_report=reports["candidate"],
        block_score_report=reports["scores"],
        input_manifest_report=reports["inputs"],
        resource_projection_report=reports["resource"],
        resource_summary=resource_summary,
        contract_sha256=str(contract["contract_sha256"]),
    )
    verify_frozen_stage0(frozen)
    ledger.freeze()
    return candidate_frame, block_scores, input_manifest, frozen


def verify_frozen_stage0(frozen: FrozenStage0) -> None:
    pairs = [
        (frozen.candidate_path, frozen.candidate_report, "candidate path"),
        (frozen.block_score_path, frozen.block_score_report, "block GR score"),
        (frozen.input_manifest_path, frozen.input_manifest_report, "input manifest"),
        (
            frozen.resource_projection_path,
            frozen.resource_projection_report,
            "resource projection",
        ),
    ]
    for path, expected, label in pairs:
        if artifact_sha_report(path)["content_sha256"] != expected["content_sha256"]:
            raise ValueError(f"{label} content changed after freeze")
    payload = json.loads(frozen.freeze_manifest_path.read_text())
    if payload["contract_sha256"] != frozen.contract_sha256:
        raise ValueError("Frozen contract SHA mismatch")


# %% [markdown]
# ## 7. Late truth and hidden-like attachment

# %%
def load_truth_late(
    wells: Sequence[str],
    train_dir: Path,
    ledger: TruthAccessLedger,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for well in sorted(set(str(value) for value in wells)):
        horizontal = pd.read_csv(
            train_dir / f"{well}__horizontal_well.csv",
            usecols=["TVT", "TVT_input"],
        )
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
    if actual_sha != str(hidden["expected_sha256"]):
        raise ValueError("Hidden-like assignment SHA mismatch")
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
        hidden_like, on="well_id", how="left", validate="many_to_one"
    )
    if merged[role_columns].isna().any().any():
        raise ValueError("Hidden-like late join left missing roles")
    stage0 = get_nested(config, "validation.stage_0") or {}
    signs = [int(value) for value in stage0["candidate_signs"]]
    zero_rank = [int(value) for value in stage0["zero_first_reference_rank"]]
    by_well = {
        well: frame.sort_values("suffix_row", kind="mergesort").reset_index(drop=True)
        for well, frame in merged.groupby("well_id", sort=True)
    }
    records: list[dict[str, Any]] = []
    for score in block_scores.itertuples(index=False):
        well = str(score.well_id)
        start = int(score.start_suffix_row)
        stop = int(score.stop_suffix_row_exclusive)
        block = by_well[well].iloc[start:stop]
        if len(block) != int(score.block_rows):
            raise ValueError(f"Block row-count mismatch for well={well}")
        true = block["true_tvt"].to_numpy(np.float64)
        mse: dict[int, float] = {}
        rmse: dict[int, float] = {}
        for sign in signs:
            prediction = block[f"path_{SIGN_COLUMN[sign]}"].to_numpy(np.float64)
            mse[sign] = float(np.mean((prediction - true) ** 2))
            rmse[sign] = float(np.sqrt(mse[sign]))
        oracle_rank = sorted(signs, key=lambda sign: (rmse[sign], zero_rank.index(sign)))
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
        records.append(row)
    result = (
        pd.DataFrame(records)
        .sort_values(["well_id", "block_index"], kind="mergesort")
        .reset_index(drop=True)
    )
    if not np.isfinite(
        result.select_dtypes(include=[np.number]).to_numpy(np.float64)
    ).all():
        raise ValueError("Post-freeze block readout contains non-finite values")
    return result


# %% [markdown]
# ## 8. Stage 0 metrics and promotion gates

# %%
def assign_group_folds(
    block_readout: pd.DataFrame,
    requested_splits: int,
) -> pd.DataFrame:
    frame = block_readout.copy()
    splits = min(int(requested_splits), int(frame["well_id"].nunique()))
    if splits < 2:
        frame["fold"] = 0
        return frame
    splitter = GroupKFold(n_splits=splits)
    fold_values = np.full(len(frame), -1, dtype=np.int16)
    groups = frame["well_id"].astype(str).to_numpy()
    for fold, (_, valid_index) in enumerate(
        splitter.split(np.zeros((len(frame), 1), dtype=np.float32), groups=groups)
    ):
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
    roles = (get_nested(config, "data.hidden_like") or {}).get("role_columns") or {}
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
    return pd.DataFrame(
        [
            {"scope": scope, **ranking_metrics(block_readout.loc[mask])}
            for scope, mask in scopes
        ]
    )


def build_fold_metrics(block_readout: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for fold, frame in block_readout.groupby("fold", sort=True):
        metric = ranking_metrics(frame)
        passed = bool(
            metric["top1_gain_vs_zero_first"] is not None
            and metric["top1_gain_vs_zero_first"] > 0.0
            and metric["mrr_gain_vs_zero_first"] > 0.0
            and metric["real_minus_circular_top1"] > 0.0
        )
        records.append({"fold": int(fold), **metric, "direction_pass": passed})
    return pd.DataFrame(records)


def evaluate_stage0_gates(
    candidate_paths: pd.DataFrame,
    block_scores: pd.DataFrame,
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
    resource = frozen.resource_summary
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
        "resource_sha_readback": (
            artifact_sha_report(frozen.resource_projection_path)["content_sha256"]
            == frozen.resource_projection_report["content_sha256"]
        ),
        "candidate_ids_unique": not candidate_paths["id"].duplicated().any(),
        "block_keys_unique": not block_scores.duplicated(
            ["well_id", "block_index"]
        ).any(),
        "complete_512_row_blocks_only": bool(block_scores["block_rows"].eq(512).all()),
        "resource_projection_has_16_wells": (
            int(resource["resource_projection_wells"]) == 16
        ),
        "resource_projection_includes_extrema": bool(
            resource["includes_minimum_workload"]
            and resource["includes_maximum_workload"]
        ),
        "stage0_exact_hmm_well_runs_zero": (
            int(get_nested(config, "validation.stage_0.hmm_well_runs")) == 0
        ),
        "stage1_disabled": not bool(get_nested(config, "execution.run_stage_1")),
    }
    direction_scopes = [
        "1000_plus",
        "verification_like_spatial",
        "verification_like_typewell_purged",
    ]
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
                    scope_lookup.loc[scope, "selected_path_rmse_gain_vs_zero_ft"]
                )
                > 0.0
            )
            for scope in direction_scopes
        },
        "projected_runtime_within_budget": float(
            resource["projected_runtime_seconds"]
        )
        <= float(gates["maximum_projected_runtime_seconds"]),
        "projected_peak_rss_within_budget": float(
            resource["projected_peak_rss_gb"]
        )
        <= float(gates["maximum_projected_peak_rss_gb"]),
    }
    passed = bool(all(technical.values()) and all(scientific.values()))
    return {
        "technical": technical,
        "scientific": scientific,
        "resource": resource,
        "technical_pass": bool(all(technical.values())),
        "scientific_pass": bool(all(scientific.values())),
        "stage0_pass": bool(not debug and passed),
        "debug_never_promotes": bool(debug),
        "decision_if_run": (
            "STAGE0_PASS_REQUEST_SEPARATE_STAGE1_APPROVAL"
            if not debug and passed
            else "STAGE0_FAIL_CLOSE_WITHOUT_RESCUE"
            if not debug
            else "DEBUG_ONLY_NO_DECISION"
        ),
    }


# %% [markdown]
# ## 9. Execution orchestration and generated artifacts

# %%
def run_stage0(
    config: dict[str, Any],
    package_dir: Path,
    maximum_wells: int | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    started = time.time()
    require_authoritative_runtime()
    contract = validate_scientific_contract(config, require_run_approval=True)
    train_dir = resolve_train_dir(config)
    artifacts_dir, _ = output_dirs(package_dir)
    ledger = TruthAccessLedger()
    print("Experiment:", EXPERIMENT_NAME)
    print("Route:", get_nested(config, "experiment.route"))
    print("Parent:", get_nested(config, "lineage.parent"))
    print("Contract SHA256:", contract["contract_sha256"])
    print("Stage 0: 3 fixed paths / 16 resource wells / exact HMM runs 0")
    print("Stage 1 planned: 1 variant / 773 exact HMM runs (disabled)")
    print("LightGBM configs / trained folds / boosters / control reruns: 0 / 0 / 0 / 0")

    candidate_paths, block_scores, input_manifest, frozen = (
        generate_and_freeze_stage0(
            train_dir,
            artifacts_dir,
            config,
            contract,
            ledger,
            maximum_wells,
        )
    )
    verify_frozen_stage0(frozen)
    truth = load_truth_late(
        candidate_paths["well_id"].unique().tolist(), train_dir, ledger
    )
    hidden_like, hidden_report = load_hidden_like_late(config, ledger)
    block_readout = build_postfreeze_block_readout(
        candidate_paths, block_scores, truth, hidden_like, config
    )
    block_readout = assign_group_folds(
        block_readout, int(get_nested(config, "validation.n_folds"))
    )
    scope_metrics = build_scope_metrics(block_readout, config)
    fold_metrics = build_fold_metrics(block_readout)
    gate_report = evaluate_stage0_gates(
        candidate_paths,
        block_scores,
        scope_metrics,
        fold_metrics,
        ledger,
        frozen,
        config,
        debug,
    )

    paths = {
        "postfreeze_block_readout": artifacts_dir
        / f"{OUTPUT_PREFIX}_postfreeze_block_readout.csv.gz",
        "scope_metrics": artifacts_dir / f"{OUTPUT_PREFIX}_scope_metrics.csv",
        "fold_metrics": artifacts_dir / f"{OUTPUT_PREFIX}_fold_metrics.csv",
        "gate_report": artifacts_dir / f"{OUTPUT_PREFIX}_gate_report.json",
        "summary": artifacts_dir / f"{OUTPUT_PREFIX}_summary.json",
    }
    write_deterministic_csv_gzip(
        block_readout, paths["postfreeze_block_readout"]
    )
    scope_metrics.to_csv(paths["scope_metrics"], index=False)
    fold_metrics.to_csv(paths["fold_metrics"], index=False)
    write_json(paths["gate_report"], gate_report)
    artifact_reports = {
        "candidate_paths": frozen.candidate_report,
        "block_gr_scores": frozen.block_score_report,
        "input_manifest": frozen.input_manifest_report,
        "resource_projection": frozen.resource_projection_report,
        "freeze_manifest": artifact_sha_report(frozen.freeze_manifest_path),
        **{
            name: artifact_sha_report(path)
            for name, path in paths.items()
            if name != "summary"
        },
    }
    overall = (
        scope_metrics.loc[scope_metrics["scope"].eq("overall")].iloc[0].to_dict()
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
        "resource_projection": frozen.resource_summary,
        "overall": overall,
        "passing_folds": int(fold_metrics["direction_pass"].sum()),
        "gates": gate_report,
        "execution_counts": get_nested(config, "execution.stage_0_counts"),
        "artifacts": artifact_reports,
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
            "metric": "rmse",
            "stage0": {
                "overall": overall,
                "passing_folds": summary["passing_folds"],
                "resource_projection": frozen.resource_summary,
                "gates": gate_report,
                "summary_path": str(paths["summary"]),
                "summary_sha256": sha256_file(paths["summary"]),
            },
            "notes": (
                "Stage 0 only; Stage 1 exact HMM, inference, blend, and "
                "submission remain disabled."
            ),
        },
    )
    print(scope_metrics.to_string(index=False))
    print(fold_metrics.to_string(index=False))
    print(json.dumps(gate_report, indent=2))
    return summary


# %% [markdown]
# ## 10. Setup and fail-closed execution selection

# %%
if __name__ == "__main__":
    PACKAGE_DIR = resolve_package_dir()
    CONFIG = load_config(PACKAGE_DIR)
    CONTRACT = validate_scientific_contract(CONFIG)
    print(
        {
            "experiment": EXPERIMENT_NAME,
            "route": get_nested(CONFIG, "experiment.route"),
            "parent": get_nested(CONFIG, "lineage.parent"),
            "status": get_nested(CONFIG, "experiment.status"),
            "contract_sha256": CONTRACT["contract_sha256"],
            "run_stage_0": get_nested(CONFIG, "execution.run_stage_0"),
            "run_stage_1": False,
            "inference_enabled": False,
            "submission_enabled": False,
        }
    )
    if not bool(get_nested(CONFIG, "execution.run_stage_0")):
        raise RuntimeError(
            "Stage 0 is implemented but execution.run_stage_0 is false. "
            "Obtain separate Kaggle package/push/run approval before enabling it."
        )
    DEBUG = os.environ.get("EXPERIMENT_DEBUG", "0") == "1"
    MAX_WELLS_ENV = os.environ.get("EXPERIMENT_MAX_WELLS")
    MAX_WELLS = int(MAX_WELLS_ENV) if MAX_WELLS_ENV else None
    STAGE0_SUMMARY = run_stage0(
        CONFIG,
        package_dir=PACKAGE_DIR,
        maximum_wells=MAX_WELLS,
        debug=DEBUG,
    )
