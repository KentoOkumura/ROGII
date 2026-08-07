# %% [markdown]
# # exp370 triggered reset rejuvenation PF — Stage 0 train-side diagnostic
#
# This notebook implements only the frozen Stage 0 diagnostic. It runs one
# exp072-compatible 500-particle likelihood PF seed per training well, records
# pre-resampling ESS, combines ESS collapse with a visible-prefix calibrated GR
# change trigger, and queries an outer-train-only horizontal-GR atlas. Target
# well TVT and hidden-like roles are attached only after all trigger, atlas,
# proposal, score, and fold artifacts have been written and SHA-verified.
#
# Stage 1 particle reinjection, inference, and submission remain unimplemented.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Notebook-safe runtime, configuration, path, and SHA helpers
# 3. Frozen Stage 0 scientific and execution contract
# 4. Fold assignment, truth-access ledger, and raw-input helpers
# 5. Exp072-compatible one-seed diagnostic likelihood PF
# 6. Fold-safe horizontal-GR atlas and top-3 proposal helpers
# 7. Target-free trigger, PF, and proposal generation
# 8. Pre-truth artifact freeze and SHA readback
# 9. Late truth and hidden-like attachment
# 10. Stage 0 metrics and promotion gates
# 11. Execution orchestration and generated artifacts
# 12. Setup and fail-closed execution selection

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    from numba import njit
except ImportError:

    def njit(*decorator_args: Any, **decorator_kwargs: Any) -> Any:
        del decorator_kwargs
        if decorator_args and callable(decorator_args[0]):
            return decorator_args[0]

        def decorate(function: Any) -> Any:
            return function

        return decorate

try:
    from IPython import get_ipython
    from IPython.display import display
except ImportError:

    def get_ipython() -> Any:
        return None

    def display(value: Any) -> None:
        print(value)


EXPERIMENT_NAME = "exp370_triggered_reset_rejuvenation_pf"
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
OUTPUT_PREFIX = f"{EXPERIMENT_NAME}_stage0"
FORBIDDEN_TARGET_PREFREEZE_COLUMNS = {
    "TVT",
    "truth",
    "target",
    "error",
    "abs_error",
    "rmse",
}

# %% [markdown]
# ## 2. Notebook-safe runtime, configuration, path, and SHA helpers

# %%
def get_nested(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def locate_config() -> Path:
    relative = Path("experiments") / EXPERIMENT_NAME / "config.yaml"
    candidates = [
        PACKAGE_DIR / "config.yaml",
        PACKAGE_DIR / relative,
        Path.cwd() / "config.yaml",
        Path.cwd() / relative,
        KAGGLE_WORKING_ROOT / "config.yaml",
    ]
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(
            path
            for path in sorted(KAGGLE_INPUT_ROOT.glob("**/config.yaml"))
            if path.parent.name == EXPERIMENT_NAME
        )
    for candidate in candidates:
        if not candidate.exists():
            continue
        value = yaml.safe_load(candidate.read_text()) or {}
        if get_nested(value, "experiment.name") == EXPERIMENT_NAME:
            return candidate
    raise FileNotFoundError(f"Could not locate config.yaml for {EXPERIMENT_NAME}")


def load_config(path: Path | None = None) -> dict[str, Any]:
    source = path or locate_config()
    if source.is_dir():
        source = source / "config.yaml"
    value = yaml.safe_load(source.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError("config.yaml must contain a mapping")
    return value


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def mapping_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_decompressed_gzip(path: str | Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_schema_sha256(frame: pd.DataFrame) -> str:
    return mapping_sha256(
        [{"column": str(column), "dtype": str(frame[column].dtype)} for column in frame]
    )


def logical_dataframe_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    for column in frame.columns:
        digest.update(str(column).encode())
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series.dtype):
            values = np.ascontiguousarray(series.to_numpy())
            digest.update(str(values.dtype).encode())
            digest.update(values.tobytes())
        else:
            for value in series.astype(str):
                digest.update(value.encode())
                digest.update(b"\n")
    return digest.hexdigest()


def stable_seed(*parts: Any, modulo: int = 2_147_483_647) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return int(value % (modulo - 1)) + 1


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(value), indent=2, ensure_ascii=False, sort_keys=True)
        + "\n"
    )


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def write_deterministic_csv_gzip(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            zipped.write(frame.to_csv(index=False, lineterminator="\n").encode())


def artifact_report(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "raw_sha256": sha256_file(path),
    }
    report["content_sha256"] = (
        sha256_decompressed_gzip(path)
        if path.suffix == ".gz"
        else report["raw_sha256"]
    )
    return report


def is_kaggle_runtime() -> bool:
    return KAGGLE_INPUT_ROOT.exists() and KAGGLE_WORKING_ROOT.exists()


def require_authoritative_runtime() -> None:
    if is_kaggle_runtime():
        return
    if os.environ.get("EXPERIMENT_ALLOW_LOCAL", "0") != "1":
        raise RuntimeError(
            "Kaggle Notebook is authoritative. Local execution requires "
            "EXPERIMENT_ALLOW_LOCAL=1 and explicit user approval."
        )


def resolve_train_dir(config: Mapping[str, Any]) -> Path:
    configured = Path(str(get_nested(config, "data.train_dir") or "data/raw/train"))
    competition_slug = str(
        get_nested(config, "runtime.kaggle.competition_slug")
        or "rogii-wellbore-geology-prediction"
    )
    expected_wells = int(
        get_nested(config, "data.parent_control.expected_wells") or 0
    )
    candidates = [
        configured,
        Path.cwd() / configured,
        PACKAGE_DIR / configured,
        Path.cwd().parent.parent / configured,
        KAGGLE_INPUT_ROOT / "competitions" / competition_slug / "train",
        KAGGLE_INPUT_ROOT / competition_slug / "train",
    ]

    def paired_well_count(candidate: Path) -> int:
        horizontal = {
            path.name.removesuffix("__horizontal_well.csv")
            for path in candidate.glob("*__horizontal_well.csv")
        }
        typewell = {
            path.name.removesuffix("__typewell.csv")
            for path in candidate.glob("*__typewell.csv")
        }
        return len(horizontal.intersection(typewell))

    for candidate in candidates:
        if not candidate.exists():
            continue
        count = paired_well_count(candidate)
        if count and (not expected_wells or count == expected_wells):
            return candidate
    if KAGGLE_INPUT_ROOT.exists():
        parents = sorted(
            {
                path.parent
                for path in KAGGLE_INPUT_ROOT.glob("**/*__horizontal_well.csv")
                if path.parent.name == "train"
            }
        )
        for parent in parents:
            count = paired_well_count(parent)
            if count and (not expected_wells or count == expected_wells):
                return parent
    raise FileNotFoundError("Could not resolve raw train directory")


def resolve_existing(
    filename: str,
    candidates: Sequence[str],
    patterns: Sequence[str],
) -> Path:
    direct = [Path(value) for value in candidates]
    for candidate in direct:
        path = candidate if candidate.name == filename else candidate / filename
        if path.exists():
            return path
    roots = [Path.cwd(), PACKAGE_DIR, KAGGLE_INPUT_ROOT]
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            found = sorted(root.glob(pattern))
            if found:
                return found[0]
    raise FileNotFoundError(f"Could not resolve required input {filename}")


def output_directory(config: Mapping[str, Any]) -> Path:
    if is_kaggle_runtime():
        path = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        config_path = locate_config()
        path = config_path.parent / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "device": "cpu",
    }


# %% [markdown]
# ## 3. Frozen Stage 0 scientific and execution contract
#
# The implementation turn may only make Stage 0 code runnable. A Kaggle run
# still requires separate approval and `execution.run_stage_0=true`.

# %%
def stage0_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    stage0 = get_nested(config, "validation.stage_0") or {}
    trigger = stage0.get("trigger") or {}
    atlas = stage0.get("atlas") or {}
    gates = stage0.get("all_required") or {}
    fixed = get_nested(config, "model.fixed_from_exp072") or {}
    execution = get_nested(config, "execution") or {}
    counts = execution.get("stage_0_counts") or {}
    contract = {
        "experiment": get_nested(config, "experiment.name"),
        "route": get_nested(config, "experiment.route"),
        "parent": get_nested(config, "lineage.parent"),
        "implementation_scope": execution.get("implementation_scope"),
        "particles": int(stage0.get("particles")),
        "diagnostic_seed_count": int(stage0.get("diagnostic_seed_count")),
        "diagnostic_pf_seed_well_runs": int(
            stage0.get("diagnostic_pf_seed_well_runs")
        ),
        "bad_event_horizon_rows": int(stage0.get("bad_event_horizon_rows")),
        "trigger": {
            "gr_change_quantile": float(
                trigger.get("gr_change_robust_z_quantile_from_known_prefix")
            ),
            "maximum_ess_fraction": float(trigger.get("maximum_ess_fraction")),
            "combine": str(trigger.get("combine")),
            "refractory_rows": int(trigger.get("refractory_rows")),
            "negative_control_shift_rows": int(
                trigger.get("negative_control_shift_rows")
            ),
        },
        "atlas": {
            "donor_scope": str(atlas.get("donor_scope")),
            "query_window_rows": int(atlas.get("query_window_rows")),
            "patch_points": int(atlas.get("patch_points")),
            "source_stride_rows": int(atlas.get("source_stride_rows")),
            "tvt_bin_width_ft": float(atlas.get("tvt_bin_width_ft")),
            "maximum_patches_per_well_bin": int(
                atlas.get("maximum_patches_per_well_bin")
            ),
            "minimum_source_wells_per_bin": int(
                atlas.get("minimum_source_wells_per_bin")
            ),
            "similarity": str(atlas.get("similarity")),
            "top_k": int(atlas.get("top_k")),
            "minimum_tvt_separation_ft": float(
                atlas.get("minimum_tvt_separation_ft")
            ),
            "stable_tie_break": str(atlas.get("stable_tie_break")),
        },
        "fixed_pf": {
            "momentum": float(fixed.get("momentum")),
            "velocity_noise": float(fixed.get("velocity_noise")),
            "position_noise": float(fixed.get("position_noise")),
            "resample_threshold": float(fixed.get("resample_threshold")),
            "resample_position_noise": float(fixed.get("resample_position_noise")),
            "resample_velocity_noise": float(fixed.get("resample_velocity_noise")),
            "initial_position_jitter_ft": float(
                fixed.get("initial_position_jitter_ft")
            ),
        },
        "gates": to_jsonable(gates),
        "run_stage_0": bool(execution.get("run_stage_0")),
        "run_stage_1": bool(execution.get("run_stage_1")),
        "run_inference": bool(execution.get("run_inference")),
        "create_submission": bool(execution.get("create_submission")),
        "stage0_counts": to_jsonable(counts),
    }
    return contract | {"contract_sha256": mapping_sha256(contract)}


def validate_scientific_contract(
    config: Mapping[str, Any],
    *,
    require_run_approval: bool = False,
) -> dict[str, Any]:
    contract = stage0_contract(config)
    expected = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "parent": "exp072_exp063_full_replay_feature_cache",
        "implementation_scope": "stage0_only",
        "particles": 500,
        "diagnostic_seed_count": 1,
        "diagnostic_pf_seed_well_runs": 773,
        "bad_event_horizon_rows": 128,
        "run_stage_1": False,
        "run_inference": False,
        "create_submission": False,
    }
    mismatches = {
        key: {"expected": value, "actual": contract.get(key)}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    trigger_expected = {
        "gr_change_quantile": 0.995,
        "maximum_ess_fraction": 0.20,
        "combine": "logical_and",
        "refractory_rows": 512,
        "negative_control_shift_rows": 512,
    }
    atlas_expected = {
        "donor_scope": "outer_train_wells_only",
        "query_window_rows": 256,
        "patch_points": 32,
        "source_stride_rows": 32,
        "tvt_bin_width_ft": 2.0,
        "maximum_patches_per_well_bin": 6,
        "minimum_source_wells_per_bin": 2,
        "similarity": "zero_mean_normalized_cross_correlation",
        "top_k": 3,
        "minimum_tvt_separation_ft": 10.0,
        "stable_tie_break": "higher_zncc_then_lower_tvt",
    }
    for label, expected_mapping in (
        ("trigger", trigger_expected),
        ("atlas", atlas_expected),
    ):
        actual = contract[label]
        for key, value in expected_mapping.items():
            if actual.get(key) != value:
                mismatches[f"{label}.{key}"] = {
                    "expected": value,
                    "actual": actual.get(key),
                }
    fixed_expected = {
        "momentum": 0.998,
        "velocity_noise": 0.002,
        "position_noise": 0.005,
        "resample_threshold": 0.5,
        "resample_position_noise": 0.10,
        "resample_velocity_noise": 0.001,
        "initial_position_jitter_ft": 4.5,
    }
    for key, value in fixed_expected.items():
        if contract["fixed_pf"].get(key) != value:
            mismatches[f"fixed_pf.{key}"] = {
                "expected": value,
                "actual": contract["fixed_pf"].get(key),
            }
    if mismatches:
        raise ValueError(f"Frozen exp370 Stage 0 contract mismatch: {mismatches}")
    if not bool(get_nested(config, "execution.implementation_approved")):
        raise PermissionError("Stage 0 implementation is not approved")
    if require_run_approval:
        if not bool(get_nested(config, "execution.kaggle_execution_approved")):
            raise PermissionError("Kaggle Stage 0 execution is not approved")
        if not bool(get_nested(config, "execution.kaggle_push_approved")):
            raise PermissionError("Kaggle Stage 0 push is not approved")
        if not contract["run_stage_0"]:
            raise PermissionError("execution.run_stage_0 is false")
    return contract


# %% [markdown]
# ## 4. Fold assignment, truth-access ledger, and raw-input helpers
#
# Outer-train donor TVT is permitted only in atlases for other folds. The
# validation well's own TVT is never requested until the SHA freeze completes.

# %%
@dataclass
class TruthAccessLedger:
    frozen: bool = False
    target_truth_rows_before_freeze: int = 0
    hidden_role_rows_before_freeze: int = 0
    outer_train_donor_truth_rows_before_freeze: int = 0
    donor_fold_leakage_violations: int = 0
    target_truth_rows_after_freeze: int = 0
    hidden_role_rows_after_freeze: int = 0

    def guard_target_prefreeze_columns(
        self,
        columns: Iterable[str],
        rows: int,
        label: str,
    ) -> None:
        overlap = FORBIDDEN_TARGET_PREFREEZE_COLUMNS.intersection(columns)
        if overlap:
            if not self.frozen:
                self.target_truth_rows_before_freeze += int(rows)
            raise ValueError(
                f"{label}: forbidden target pre-freeze columns requested: "
                f"{sorted(overlap)}"
            )

    def record_outer_train_donor_truth(
        self,
        *,
        donor_fold: int,
        atlas_fold: int,
        rows: int,
    ) -> None:
        if donor_fold == atlas_fold:
            self.donor_fold_leakage_violations += 1
            raise RuntimeError(
                f"fold={atlas_fold} atlas attempted to use validation-fold donor truth"
            )
        self.outer_train_donor_truth_rows_before_freeze += int(rows)

    def freeze(self) -> None:
        if (
            self.target_truth_rows_before_freeze
            or self.hidden_role_rows_before_freeze
            or self.donor_fold_leakage_violations
        ):
            raise RuntimeError("target truth, hidden role, or donor-fold leakage before SHA freeze")
        self.frozen = True

    def record_target_truth_late(self, rows: int) -> None:
        if not self.frozen:
            self.target_truth_rows_before_freeze += int(rows)
            raise RuntimeError("target suffix truth cannot be read before SHA freeze")
        self.target_truth_rows_after_freeze += int(rows)

    def record_hidden_late(self, rows: int) -> None:
        if not self.frozen:
            self.hidden_role_rows_before_freeze += int(rows)
            raise RuntimeError("hidden-like roles cannot be read before SHA freeze")
        self.hidden_role_rows_after_freeze += int(rows)


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


def load_saved_parent_likpf(
    config: Mapping[str, Any],
    maximum_wells: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.parent_control") or {}
    filename = str(spec["filename"])
    path = resolve_existing(
        filename,
        [str(value) for value in spec.get("candidates", [])],
        [f"**/{filename}"],
    )
    report = artifact_report(path)
    if report["content_sha256"] != str(spec["expected_decompressed_sha256"]):
        raise ValueError("Saved exp072 likPF decompressed SHA mismatch")
    safe_columns = [str(value) for value in spec["safe_columns"]]
    frame = pd.read_csv(
        path,
        usecols=safe_columns,
        dtype={"id": str, "well": str},
    )
    frame = frame.rename(columns={"well": "well_id"})
    frame["row_idx"] = pd.to_numeric(
        frame["id"].str.rsplit("_", n=1).str[-1],
        errors="raise",
    ).astype(np.int64)
    anchor = pd.to_numeric(
        frame[str(spec["anchor_column"])], errors="raise"
    ).to_numpy(np.float64)
    delta = pd.to_numeric(
        frame[str(spec["delta_column"])], errors="raise"
    ).to_numpy(np.float64)
    frame["saved_likpf_tvt"] = anchor + delta
    frame = frame[
        ["id", "well_id", "row_idx", "saved_likpf_tvt"]
    ].sort_values(["well_id", "row_idx"], kind="mergesort")
    if frame.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("Saved exp072 likPF row identity is duplicated")
    if not np.isfinite(frame["saved_likpf_tvt"].to_numpy(np.float64)).all():
        raise ValueError("Saved exp072 likPF contains non-finite predictions")
    if maximum_wells is None:
        if (
            len(frame) != int(spec["expected_rows"])
            or frame["well_id"].nunique() != int(spec["expected_wells"])
        ):
            raise ValueError("Saved exp072 likPF row/well coverage mismatch")
    else:
        selected = sorted(frame["well_id"].unique().tolist())[: int(maximum_wells)]
        frame = frame.loc[frame["well_id"].isin(selected)].reset_index(drop=True)
    return frame.reset_index(drop=True), {
        **report,
        "rows": int(len(frame)),
        "wells": int(frame["well_id"].nunique()),
        "safe_columns": safe_columns,
        "prediction_representation": "last_known_tvt_plus_likpf_mean_d",
    }


def deterministic_well_folds(
    wells: Sequence[str],
    n_folds: int,
    seed: int,
) -> dict[str, int]:
    ordered = np.asarray(sorted(map(str, wells)), dtype=object)
    if n_folds < 2 or len(ordered) < n_folds:
        raise ValueError("Fold-safe atlas needs at least n_folds wells and n_folds >= 2")
    shuffled = ordered.copy()
    np.random.default_rng(int(seed)).shuffle(shuffled)
    assignment: dict[str, int] = {}
    for fold, values in enumerate(np.array_split(shuffled, int(n_folds))):
        for well in values.tolist():
            assignment[str(well)] = int(fold)
    if sorted(assignment) != sorted(map(str, wells)):
        raise RuntimeError("Fold assignment does not cover every well exactly once")
    return assignment


def load_target_prefreeze_well(
    well: str,
    train_dir: Path,
    ledger: TruthAccessLedger,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    horizontal_path = train_dir / f"{well}__horizontal_well.csv"
    typewell_path = train_dir / f"{well}__typewell.csv"
    horizontal_columns = ["MD", "Z", "GR", "TVT_input"]
    ledger.guard_target_prefreeze_columns(horizontal_columns, 0, f"{well} target")
    horizontal = pd.read_csv(horizontal_path, usecols=horizontal_columns)
    typewell = pd.read_csv(typewell_path, usecols=["TVT", "GR"])
    if horizontal.empty or typewell.empty:
        raise ValueError(f"Empty raw input for well={well}")
    return horizontal, typewell, horizontal_path, typewell_path


def prepare_typewell(
    typewell: pd.DataFrame,
    grid_step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    clean = typewell.copy()
    clean["TVT"] = pd.to_numeric(clean["TVT"], errors="coerce")
    clean["GR"] = pd.to_numeric(clean["GR"], errors="coerce")
    clean = (
        clean.dropna(subset=["TVT", "GR"])
        .sort_values("TVT", kind="mergesort")
        .drop_duplicates("TVT", keep="last")
    )
    if len(clean) < 3:
        raise ValueError("Typewell needs at least three finite TVT/GR rows")
    tvt = clean["TVT"].to_numpy(np.float64)
    gr = clean["GR"].to_numpy(np.float64)
    grid_min = float(tvt.min())
    grid = np.arange(grid_min, float(tvt.max()) + float(grid_step), float(grid_step))
    grid_gr = np.interp(grid, tvt, gr).astype(np.float64)
    return tvt, gr, grid_gr, grid_min, float(grid_step)


def interpolate_observed_gr(values: pd.Series, fallback: float) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    return (
        numeric.interpolate(limit_direction="both")
        .fillna(float(fallback))
        .to_numpy(np.float64)
    )


def exp072_gr_sigma(
    horizontal: pd.DataFrame,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
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
    residual = observed - np.interp(known_tvt, typewell_tvt, typewell_gr)
    sigma = float(np.std(residual))
    return float(np.clip(sigma, float(sigma_min), float(sigma_max)))


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
    delta_tvt = np.diff(tvt)
    delta_z = np.diff(z)
    delta_md = np.diff(md)
    valid = (
        np.isfinite(delta_tvt)
        & np.isfinite(delta_z)
        & np.isfinite(delta_md)
        & (delta_md > 0.0)
    )
    count = int(valid.sum())
    if count < int(minimum_valid_steps):
        return float(fallback), count
    return float(np.median((delta_tvt[valid] + delta_z[valid]) / delta_md[valid])), count


def robust_location_scale(
    values: np.ndarray,
    floor: float = 1.0e-6,
) -> tuple[float, float]:
    finite = np.asarray(values, np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 4:
        raise ValueError("At least four finite prefix values are required")
    location = float(np.median(finite))
    scale = float(1.4826 * np.median(np.abs(finite - location)))
    if not np.isfinite(scale) or scale < floor:
        scale = float(np.std(finite))
    return location, max(scale, float(floor))


def apply_refractory(candidate_mask: np.ndarray, refractory_rows: int) -> np.ndarray:
    candidate = np.asarray(candidate_mask, dtype=bool)
    accepted = np.zeros(len(candidate), dtype=bool)
    next_allowed = 0
    for index in np.flatnonzero(candidate):
        if int(index) < next_allowed:
            continue
        accepted[int(index)] = True
        next_allowed = int(index) + int(refractory_rows)
    return accepted


def circular_shift_trigger_score(values: np.ndarray, shift_rows: int) -> np.ndarray:
    score = np.asarray(values, np.float64)
    if len(score) <= 1:
        return np.zeros_like(score)
    shift = int(shift_rows) % len(score)
    if shift == 0:
        shift = 1
    return np.roll(score, shift)


# %% [markdown]
# ## 5. Exp072-compatible one-seed diagnostic likelihood PF
#
# ESS is captured after the GR likelihood update and before any systematic
# resampling. The baseline prediction itself follows exp072's update order.

# %%
@njit(cache=True, nogil=True)
def _interp_uniform_grid(
    grid_values: np.ndarray,
    value: float,
    grid_min: float,
    grid_step: float,
) -> float:
    coordinate = (value - grid_min) / grid_step
    left = int(math.floor(coordinate))
    fraction = coordinate - left
    if left < 0:
        left = 0
        fraction = 0.0
    if left >= len(grid_values) - 1:
        return float(grid_values[-1])
    return float(
        grid_values[left] * (1.0 - fraction)
        + grid_values[left + 1] * fraction
    )


@njit(cache=True, nogil=True)
def diagnostic_likelihood_pf(
    md: np.ndarray,
    z: np.ndarray,
    observed_gr: np.ndarray,
    typewell_grid_gr: np.ndarray,
    grid_min: float,
    grid_step: float,
    gr_sigma: float,
    last_known_md: float,
    last_known_position: float,
    initial_rate: float,
    particles: int,
    seed: int,
    momentum: float,
    velocity_noise: float,
    position_noise: float,
    resample_position_noise: float,
    resample_velocity_noise: float,
    resample_threshold: float,
    initial_position_jitter: float,
    support_pad: float,
) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(seed)
    rows = len(md)
    prediction = np.empty(rows, dtype=np.float64)
    ess_fraction = np.empty(rows, dtype=np.float64)
    position = np.empty(particles, dtype=np.float64)
    rate = np.empty(particles, dtype=np.float64)
    weight = np.full(particles, 1.0 / particles, dtype=np.float64)
    for particle in range(particles):
        position[particle] = last_known_position + initial_position_jitter * np.random.randn()
        rate[particle] = initial_rate + 0.01 * np.random.randn()
    previous_md = last_known_md
    grid_max = grid_min + len(typewell_grid_gr) * grid_step
    for row in range(rows):
        delta_md = md[row] - previous_md
        if delta_md < 1.0:
            delta_md = 1.0
        for particle in range(particles):
            rate[particle] = (
                momentum * rate[particle] + velocity_noise * np.random.randn()
            )
            position[particle] += (
                rate[particle] * delta_md + position_noise * np.random.randn()
            )
            tvt = position[particle] - z[row]
            if tvt < grid_min - support_pad:
                tvt = grid_min - support_pad
            if tvt > grid_max + support_pad:
                tvt = grid_max + support_pad
            position[particle] = tvt + z[row]
        weight_sum = 0.0
        for particle in range(particles):
            expected_gr = _interp_uniform_grid(
                typewell_grid_gr,
                position[particle] - z[row],
                grid_min,
                grid_step,
            )
            residual = (observed_gr[row] - expected_gr) / gr_sigma
            squared = residual * residual
            if squared > 600.0:
                squared = 600.0
            likelihood = math.exp(-0.5 * squared)
            if likelihood < 1.0e-300:
                likelihood = 1.0e-300
            weight[particle] *= likelihood
            weight_sum += weight[particle]
        if weight_sum > 0.0:
            for particle in range(particles):
                weight[particle] /= weight_sum
        else:
            for particle in range(particles):
                weight[particle] = 1.0 / particles
        inverse_ess = 0.0
        for particle in range(particles):
            inverse_ess += weight[particle] * weight[particle]
        ess = 1.0 / inverse_ess
        ess_fraction[row] = ess / particles
        if ess < resample_threshold * particles:
            cumulative = np.empty(particles, dtype=np.float64)
            total = 0.0
            for particle in range(particles):
                total += weight[particle]
                cumulative[particle] = total
            new_position = np.empty(particles, dtype=np.float64)
            new_rate = np.empty(particles, dtype=np.float64)
            cursor = 0
            start = np.random.uniform(0.0, 1.0 / particles)
            for particle in range(particles):
                target = start + particle / particles
                while cursor < particles - 1 and cumulative[cursor] < target:
                    cursor += 1
                new_position[particle] = (
                    position[cursor] + resample_position_noise * np.random.randn()
                )
                new_rate[particle] = (
                    rate[cursor] + resample_velocity_noise * np.random.randn()
                )
            for particle in range(particles):
                position[particle] = new_position[particle]
                rate[particle] = new_rate[particle]
                weight[particle] = 1.0 / particles
        estimate = 0.0
        for particle in range(particles):
            estimate += weight[particle] * (position[particle] - z[row])
        prediction[row] = estimate
        previous_md = md[row]
    return prediction, ess_fraction


# %% [markdown]
# ## 6. Fold-safe horizontal-GR atlas and top-3 proposal helpers
#
# The atlas is global within each outer-train fold. Each 256-row donor patch is
# z-normalized, resampled to 32 points, accumulated in a 2-ft TVT bin, and then
# normalized again as one prototype. A donor well contributes to four atlases
# and is excluded from the atlas where it is a validation well.

# %%
@dataclass
class AtlasAccumulator:
    patch_sum: np.ndarray
    patch_count: int = 0
    source_wells: set[str] = field(default_factory=set)


def resampled_znorm_patch(
    values: np.ndarray,
    center: int,
    window_rows: int,
    patch_points: int,
) -> np.ndarray | None:
    source = np.asarray(values, np.float64)
    half = int(window_rows) // 2
    start = int(center) - half
    stop = start + int(window_rows)
    if start < 0 or stop > len(source):
        return None
    patch = source[start:stop]
    if not np.isfinite(patch).all() or len(patch) != int(window_rows):
        return None
    source_x = np.linspace(0.0, 1.0, num=len(patch), dtype=np.float64)
    target_x = np.linspace(0.0, 1.0, num=int(patch_points), dtype=np.float64)
    resampled = np.interp(target_x, source_x, patch)
    centered = resampled - float(np.mean(resampled))
    scale = float(np.sqrt(np.mean(centered * centered)))
    if not np.isfinite(scale) or scale < 1.0e-8:
        return None
    return (centered / scale).astype(np.float32)


def normalize_prototype(values: np.ndarray) -> np.ndarray | None:
    prototype = np.asarray(values, np.float64)
    centered = prototype - float(np.mean(prototype))
    scale = float(np.sqrt(np.mean(centered * centered)))
    if not np.isfinite(scale) or scale < 1.0e-8:
        return None
    return (centered / scale).astype(np.float32)


def atlas_patch_columns(patch_points: int) -> list[str]:
    return [f"patch_{index:03d}" for index in range(int(patch_points))]


def build_fold_safe_atlases(
    wells: Sequence[str],
    fold_by_well: Mapping[str, int],
    train_dir: Path,
    config: Mapping[str, Any],
    ledger: TruthAccessLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    stage0 = get_nested(config, "validation.stage_0") or {}
    atlas_spec = stage0.get("atlas") or {}
    n_folds = int(get_nested(config, "validation.n_folds"))
    window_rows = int(atlas_spec["query_window_rows"])
    patch_points = int(atlas_spec["patch_points"])
    stride = int(atlas_spec["source_stride_rows"])
    bin_width = float(atlas_spec["tvt_bin_width_ft"])
    max_per_well_bin = int(atlas_spec["maximum_patches_per_well_bin"])
    min_source_wells = int(atlas_spec["minimum_source_wells_per_bin"])
    accumulators: list[dict[int, AtlasAccumulator]] = [
        {} for _ in range(n_folds)
    ]
    source_wells_by_fold = [set() for _ in range(n_folds)]
    validation_wells_by_fold = [
        {well for well in wells if int(fold_by_well[well]) == fold}
        for fold in range(n_folds)
    ]
    half = window_rows // 2
    raw_identity_rows: list[dict[str, Any]] = []
    for index, well in enumerate(sorted(wells), start=1):
        donor_fold = int(fold_by_well[well])
        horizontal_path = train_dir / f"{well}__horizontal_well.csv"
        donor = pd.read_csv(horizontal_path, usecols=["GR", "TVT"])
        gr_numeric = pd.to_numeric(donor["GR"], errors="coerce")
        fallback = float(gr_numeric.median()) if gr_numeric.notna().any() else 0.0
        gr = (
            gr_numeric.interpolate(limit_direction="both")
            .fillna(fallback)
            .to_numpy(np.float64)
        )
        tvt = pd.to_numeric(donor["TVT"], errors="coerce").to_numpy(np.float64)
        used_by_bin: dict[int, int] = {}
        donor_patches: list[tuple[int, np.ndarray]] = []
        for center in range(half, len(gr) - (window_rows - half) + 1, stride):
            if not np.isfinite(tvt[center]):
                continue
            bin_key = int(math.floor(float(tvt[center]) / bin_width))
            if used_by_bin.get(bin_key, 0) >= max_per_well_bin:
                continue
            patch = resampled_znorm_patch(
                gr,
                center,
                window_rows,
                patch_points,
            )
            if patch is None:
                continue
            donor_patches.append((bin_key, patch))
            used_by_bin[bin_key] = used_by_bin.get(bin_key, 0) + 1
        for atlas_fold in range(n_folds):
            if atlas_fold == donor_fold:
                continue
            ledger.record_outer_train_donor_truth(
                donor_fold=donor_fold,
                atlas_fold=atlas_fold,
                rows=len(donor),
            )
            source_wells_by_fold[atlas_fold].add(well)
            entries = accumulators[atlas_fold]
            for bin_key, patch in donor_patches:
                entry = entries.get(bin_key)
                if entry is None:
                    entry = AtlasAccumulator(
                        patch_sum=np.zeros(patch_points, dtype=np.float64)
                    )
                    entries[bin_key] = entry
                entry.patch_sum += patch
                entry.patch_count += 1
                entry.source_wells.add(well)
        raw_identity_rows.append(
            {
                "well_id": well,
                "fold": donor_fold,
                "horizontal_raw_sha256": sha256_file(horizontal_path),
                "raw_rows": int(len(donor)),
                "atlas_source_patches": int(len(donor_patches)),
            }
        )
        if index % 50 == 0 or index == len(wells):
            print(
                f"atlas donor [{index}/{len(wells)}] well={well} "
                f"patches={len(donor_patches)}",
                flush=True,
            )

    patch_columns = atlas_patch_columns(patch_points)
    rows: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []
    for fold, entries in enumerate(accumulators):
        retained_bins = 0
        retained_patches = 0
        for bin_key in sorted(entries):
            entry = entries[bin_key]
            if len(entry.source_wells) < min_source_wells:
                continue
            prototype = normalize_prototype(entry.patch_sum / entry.patch_count)
            if prototype is None:
                continue
            row: dict[str, Any] = {
                "fold": int(fold),
                "tvt_bin": int(bin_key),
                "proposal_tvt": (float(bin_key) + 0.5) * bin_width,
                "patch_count": int(entry.patch_count),
                "source_well_count": int(len(entry.source_wells)),
                "source_wells_sha256": mapping_sha256(sorted(entry.source_wells)),
            }
            row.update(
                {
                    column: float(prototype[index])
                    for index, column in enumerate(patch_columns)
                }
            )
            rows.append(row)
            retained_bins += 1
            retained_patches += int(entry.patch_count)
        valid = validation_wells_by_fold[fold]
        sources = source_wells_by_fold[fold]
        intersection = sorted(valid.intersection(sources))
        fold_reports.append(
            {
                "fold": int(fold),
                "validation_well_count": int(len(valid)),
                "source_well_count": int(len(sources)),
                "validation_source_intersection_count": int(len(intersection)),
                "validation_wells_sha256": mapping_sha256(sorted(valid)),
                "source_wells_sha256": mapping_sha256(sorted(sources)),
                "retained_tvt_bins": retained_bins,
                "retained_source_patches": retained_patches,
            }
        )
        if intersection:
            ledger.donor_fold_leakage_violations += len(intersection)
            raise RuntimeError(
                f"fold={fold} atlas contains validation wells: {intersection[:5]}"
            )
    atlas_frame = pd.DataFrame(rows)
    if atlas_frame.empty:
        atlas_frame = pd.DataFrame(
            columns=[
                "fold",
                "tvt_bin",
                "proposal_tvt",
                "patch_count",
                "source_well_count",
                "source_wells_sha256",
                *patch_columns,
            ]
        )
    else:
        atlas_frame = atlas_frame.sort_values(
            ["fold", "tvt_bin"], kind="mergesort"
        ).reset_index(drop=True)
    identity = pd.DataFrame(raw_identity_rows).sort_values(
        "well_id", kind="mergesort"
    )
    report = {
        "folds": fold_reports,
        "fold_assignment_sha256": mapping_sha256(
            [{"well_id": well, "fold": int(fold_by_well[well])} for well in sorted(wells)]
        ),
        "donor_raw_identity_sha256": logical_dataframe_sha256(identity),
        "donor_raw_identity_rows": identity.to_dict(orient="records"),
        "window_rows": window_rows,
        "patch_points": patch_points,
        "source_stride_rows": stride,
        "tvt_bin_width_ft": bin_width,
        "maximum_patches_per_well_bin": max_per_well_bin,
        "minimum_source_wells_per_bin": min_source_wells,
        "atlas_rows": int(len(atlas_frame)),
        "atlas_schema_sha256": dataframe_schema_sha256(atlas_frame),
    }
    return atlas_frame, report


def select_topk_separated(
    proposal_tvt: np.ndarray,
    score: np.ndarray,
    top_k: int,
    minimum_separation_ft: float,
) -> np.ndarray:
    tvt = np.asarray(proposal_tvt, np.float64)
    values = np.asarray(score, np.float64)
    if len(tvt) != len(values):
        raise ValueError("proposal_tvt and score must be aligned")
    order = np.lexsort((tvt, -values))
    selected: list[int] = []
    for index in order.tolist():
        if not np.isfinite(values[index]) or not np.isfinite(tvt[index]):
            continue
        if any(
            abs(float(tvt[index]) - float(tvt[chosen]))
            < float(minimum_separation_ft)
            for chosen in selected
        ):
            continue
        selected.append(int(index))
        if len(selected) == int(top_k):
            break
    return np.asarray(selected, dtype=np.int64)


def query_atlas_top3(
    full_gr: np.ndarray,
    center_row: int,
    fold_atlas: pd.DataFrame,
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    atlas_spec = get_nested(config, "validation.stage_0.atlas") or {}
    patch_points = int(atlas_spec["patch_points"])
    query = resampled_znorm_patch(
        full_gr,
        int(center_row),
        int(atlas_spec["query_window_rows"]),
        patch_points,
    )
    if query is None or fold_atlas.empty:
        return []
    patch = fold_atlas[atlas_patch_columns(patch_points)].to_numpy(np.float64)
    zncc = patch @ query.astype(np.float64) / float(patch_points)
    selected = select_topk_separated(
        fold_atlas["proposal_tvt"].to_numpy(np.float64),
        zncc,
        int(atlas_spec["top_k"]),
        float(atlas_spec["minimum_tvt_separation_ft"]),
    )
    proposals: list[dict[str, Any]] = []
    for rank, index in enumerate(selected.tolist(), start=1):
        source = fold_atlas.iloc[index]
        proposals.append(
            {
                "proposal_rank": int(rank),
                "proposal_tvt": float(source["proposal_tvt"]),
                "zncc": float(zncc[index]),
                "tvt_bin": int(source["tvt_bin"]),
                "patch_count": int(source["patch_count"]),
                "source_well_count": int(source["source_well_count"]),
                "source_wells_sha256": str(source["source_wells_sha256"]),
            }
        )
    return proposals


# %% [markdown]
# ## 7. Target-free trigger, PF, and proposal generation

# %%
def build_prefreeze_rows_for_well(
    well: str,
    fold: int,
    saved_likpf_well: pd.DataFrame,
    train_dir: Path,
    fold_atlas: pd.DataFrame,
    config: Mapping[str, Any],
    ledger: TruthAccessLedger,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    horizontal, typewell, horizontal_path, typewell_path = load_target_prefreeze_well(
        well,
        train_dir,
        ledger,
    )
    tvt_input = pd.to_numeric(
        horizontal["TVT_input"], errors="coerce"
    ).to_numpy(np.float64)
    known_mask = np.isfinite(tvt_input)
    known_index = np.flatnonzero(known_mask)
    suffix_index = np.flatnonzero(~known_mask)
    if (
        len(known_index) < 4
        or len(suffix_index) == 0
        or not np.array_equal(known_index, np.arange(len(known_index)))
    ):
        raise ValueError(f"well={well} lacks one contiguous visible prefix and suffix")
    saved = saved_likpf_well.sort_values("row_idx", kind="mergesort").reset_index(
        drop=True
    )
    if not np.array_equal(saved["row_idx"].to_numpy(np.int64), suffix_index):
        raise ValueError(f"well={well} saved exp072 likPF row identity mismatch")

    stage0 = get_nested(config, "validation.stage_0") or {}
    trigger_spec = stage0.get("trigger") or {}
    atlas_spec = stage0.get("atlas") or {}
    fixed = get_nested(config, "model.fixed_from_exp072") or {}
    typewell_tvt, typewell_gr, grid_gr, grid_min, grid_step = prepare_typewell(
        typewell,
        float(fixed["typewell_grid_step_ft"]),
    )
    full_gr = interpolate_observed_gr(
        horizontal["GR"],
        fallback=float(np.mean(typewell_gr)),
    )
    raw_gr = pd.to_numeric(horizontal["GR"], errors="coerce").to_numpy(np.float64)
    gr_sigma = exp072_gr_sigma(
        horizontal,
        typewell_tvt,
        typewell_gr,
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
    last_known_row = int(known_index[-1])
    last_known = horizontal.iloc[last_known_row]
    md = pd.to_numeric(horizontal["MD"], errors="raise").to_numpy(np.float64)
    z = pd.to_numeric(horizontal["Z"], errors="raise").to_numpy(np.float64)
    seed = stable_seed(
        EXPERIMENT_NAME,
        int(fold),
        well,
        "stage0_exp072_likpf",
        0,
    )
    diagnostic_prediction, ess_fraction = diagnostic_likelihood_pf(
        md[suffix_index],
        z[suffix_index],
        full_gr[suffix_index],
        grid_gr,
        grid_min,
        grid_step,
        gr_sigma,
        float(last_known["MD"]),
        float(last_known["TVT_input"]) + float(last_known["Z"]),
        initial_rate,
        int(stage0["particles"]),
        seed,
        float(fixed["momentum"]),
        float(fixed["velocity_noise"]),
        float(fixed["position_noise"]),
        float(fixed["resample_position_noise"]),
        float(fixed["resample_velocity_noise"]),
        float(fixed["resample_threshold"]),
        float(fixed["initial_position_jitter_ft"]),
        float(fixed["typewell_support_pad_ft"]),
    )
    if not np.isfinite(diagnostic_prediction).all() or not np.isfinite(
        ess_fraction
    ).all():
        raise ValueError(f"well={well} PF returned non-finite values")
    saved_base_prediction = saved["saved_likpf_tvt"].to_numpy(np.float64)

    absolute_change = np.full(len(horizontal), np.nan, dtype=np.float64)
    adjacent = np.isfinite(raw_gr[1:]) & np.isfinite(raw_gr[:-1])
    absolute_change[1:] = np.where(
        adjacent,
        np.abs(raw_gr[1:] - raw_gr[:-1]),
        np.nan,
    )
    change_location, change_scale = robust_location_scale(
        absolute_change[known_mask]
    )
    change_z = (absolute_change - change_location) / change_scale
    known_change_z = change_z[known_mask & np.isfinite(change_z)]
    change_threshold = float(
        np.quantile(
            known_change_z,
            float(trigger_spec["gr_change_robust_z_quantile_from_known_prefix"]),
        )
    )
    change_threshold = max(change_threshold, 1.0e-12)
    suffix_change_z = change_z[suffix_index]
    raw_observed = np.isfinite(raw_gr[suffix_index])
    ess_threshold = float(trigger_spec["maximum_ess_fraction"])
    change_ratio = suffix_change_z / change_threshold
    ess_ratio = ess_threshold / np.maximum(ess_fraction, 1.0e-12)
    trigger_strength = np.minimum(change_ratio, ess_ratio)
    trigger_strength[~np.isfinite(trigger_strength)] = 0.0
    trigger_strength[~raw_observed] = 0.0

    horizon = int(stage0["bad_event_horizon_rows"])
    window_rows = int(atlas_spec["query_window_rows"])
    half = window_rows // 2
    query_start = suffix_index - half
    query_stop = query_start + window_rows
    eligible = (
        (np.arange(len(suffix_index)) + horizon <= len(suffix_index))
        & (query_start >= 0)
        & (query_stop <= len(horizontal))
    )
    candidate = (
        eligible
        & raw_observed
        & (suffix_change_z >= change_threshold)
        & (ess_fraction <= ess_threshold)
    )
    accepted = apply_refractory(
        candidate,
        int(trigger_spec["refractory_rows"]),
    )
    accepted_score = np.where(accepted, trigger_strength, 0.0)
    circular_score = np.zeros(len(suffix_index), dtype=np.float64)
    eligible_index = np.flatnonzero(eligible)
    circular_score[eligible_index] = circular_shift_trigger_score(
        accepted_score[eligible_index],
        int(trigger_spec["negative_control_shift_rows"]),
    )
    md_since = md[suffix_index] - float(last_known["MD"])
    trigger_frame = pd.DataFrame(
        {
            "id": [f"{well}_{int(row)}" for row in suffix_index],
            "well_id": well,
            "fold": int(fold),
            "row_idx": suffix_index.astype(np.int32),
            "suffix_offset": np.arange(len(suffix_index), dtype=np.int32),
            "md_since": md_since.astype(np.float32),
            "eligible": eligible,
            "saved_likpf_tvt": saved_base_prediction.astype(np.float32),
            "diagnostic_seed_tvt": diagnostic_prediction.astype(np.float32),
            "ess_fraction": ess_fraction.astype(np.float32),
            "gr_change_robust_z": suffix_change_z.astype(np.float32),
            "gr_change_threshold_q995": float(change_threshold),
            "trigger_strength": trigger_strength.astype(np.float32),
            "accepted_trigger": accepted,
            "trigger_score": accepted_score.astype(np.float32),
            "circular_trigger_score": circular_score.astype(np.float32),
            "raw_gr_observed": raw_observed,
        }
    )

    proposal_rows: list[dict[str, Any]] = []
    for suffix_offset in np.flatnonzero(accepted):
        row_idx = int(suffix_index[suffix_offset])
        event_id = f"{well}_{row_idx}"
        proposals = query_atlas_top3(
            full_gr,
            row_idx,
            fold_atlas,
            config,
        )
        for proposal in proposals:
            proposal_rows.append(
                {
                    "event_id": event_id,
                    "well_id": well,
                    "fold": int(fold),
                    "trigger_row_idx": row_idx,
                    "trigger_suffix_offset": int(suffix_offset),
                    **proposal,
                }
            )
    proposal_frame = pd.DataFrame(proposal_rows)
    if proposal_frame.empty:
        proposal_frame = pd.DataFrame(
            columns=[
                "event_id",
                "well_id",
                "fold",
                "trigger_row_idx",
                "trigger_suffix_offset",
                "proposal_rank",
                "proposal_tvt",
                "zncc",
                "tvt_bin",
                "patch_count",
                "source_well_count",
                "source_wells_sha256",
            ]
        )
    manifest = {
        "well_id": well,
        "fold": int(fold),
        "status": "ok",
        "prefix_rows": int(len(known_index)),
        "suffix_rows": int(len(suffix_index)),
        "eligible_trigger_rows": int(eligible.sum()),
        "candidate_trigger_rows": int(candidate.sum()),
        "accepted_trigger_rows": int(accepted.sum()),
        "proposal_rows": int(len(proposal_frame)),
        "pf_seed_well_runs": 1,
        "pf_seed": int(seed),
        "particles": int(stage0["particles"]),
        "gr_sigma": float(gr_sigma),
        "initial_rate": float(initial_rate),
        "initial_rate_valid_steps": int(valid_rate_steps),
        "change_location": float(change_location),
        "change_scale": float(change_scale),
        "change_threshold_q995": float(change_threshold),
        "horizontal_raw_sha256": sha256_file(horizontal_path),
        "typewell_raw_sha256": sha256_file(typewell_path),
    }
    return trigger_frame, proposal_frame, manifest


# %% [markdown]
# ## 8. Pre-truth artifact freeze and SHA readback

# %%
@dataclass(frozen=True)
class FrozenStage0:
    fold_assignment_path: Path
    atlas_path: Path
    atlas_manifest_path: Path
    trigger_path: Path
    proposal_path: Path
    input_manifest_path: Path
    parent_control_manifest_path: Path
    freeze_manifest_path: Path
    reports: dict[str, dict[str, Any]]
    contract_sha256: str


def freeze_prefreeze_artifacts(
    output_dir: Path,
    fold_assignment: pd.DataFrame,
    atlas_frame: pd.DataFrame,
    atlas_manifest: Mapping[str, Any],
    trigger_ledger: pd.DataFrame,
    proposal_ledger: pd.DataFrame,
    input_manifest: pd.DataFrame,
    parent_control_manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
    ledger: TruthAccessLedger,
) -> FrozenStage0:
    paths = {
        "fold_assignment": output_dir / f"{OUTPUT_PREFIX}_fold_assignment.csv",
        "atlas_prototypes": output_dir / f"{OUTPUT_PREFIX}_atlas_prototypes.csv.gz",
        "atlas_manifest": output_dir / f"{OUTPUT_PREFIX}_atlas_manifest.json",
        "trigger_ledger": output_dir / f"{OUTPUT_PREFIX}_trigger_ledger.csv.gz",
        "proposal_ledger": output_dir / f"{OUTPUT_PREFIX}_proposal_ledger.csv.gz",
        "input_manifest": output_dir / f"{OUTPUT_PREFIX}_input_manifest.csv",
        "parent_control_manifest": (
            output_dir / f"{OUTPUT_PREFIX}_parent_control_manifest.json"
        ),
        "freeze_manifest": output_dir / f"{OUTPUT_PREFIX}_freeze_manifest.json",
    }
    write_csv(paths["fold_assignment"], fold_assignment)
    write_deterministic_csv_gzip(paths["atlas_prototypes"], atlas_frame)
    write_json(paths["atlas_manifest"], atlas_manifest)
    write_deterministic_csv_gzip(paths["trigger_ledger"], trigger_ledger)
    write_deterministic_csv_gzip(paths["proposal_ledger"], proposal_ledger)
    write_csv(paths["input_manifest"], input_manifest)
    write_json(paths["parent_control_manifest"], parent_control_manifest)
    reports = {
        name: artifact_report(path)
        for name, path in paths.items()
        if name != "freeze_manifest"
    }
    freeze_manifest = {
        "experiment": EXPERIMENT_NAME,
        "contract_sha256": str(contract["contract_sha256"]),
        "frozen_before_target_truth": [
            "fold_assignment",
            "atlas_prototypes",
            "atlas_manifest",
            "trigger_ledger",
            "proposal_ledger",
            "input_manifest",
            "parent_control_manifest",
        ],
        "reports": reports,
        "truth_access": to_jsonable(ledger.__dict__),
    }
    write_json(paths["freeze_manifest"], freeze_manifest)
    for name, report in reports.items():
        actual = artifact_report(paths[name])
        if actual["content_sha256"] != report["content_sha256"]:
            raise RuntimeError(f"{name} SHA readback mismatch")
    loaded_trigger = pd.read_csv(paths["trigger_ledger"])
    loaded_proposal = pd.read_csv(paths["proposal_ledger"])
    if len(loaded_trigger) != len(trigger_ledger):
        raise RuntimeError("Trigger ledger row count changed during freeze")
    if len(loaded_proposal) != len(proposal_ledger):
        raise RuntimeError("Proposal ledger row count changed during freeze")
    ledger.freeze()
    return FrozenStage0(
        fold_assignment_path=paths["fold_assignment"],
        atlas_path=paths["atlas_prototypes"],
        atlas_manifest_path=paths["atlas_manifest"],
        trigger_path=paths["trigger_ledger"],
        proposal_path=paths["proposal_ledger"],
        input_manifest_path=paths["input_manifest"],
        parent_control_manifest_path=paths["parent_control_manifest"],
        freeze_manifest_path=paths["freeze_manifest"],
        reports=reports,
        contract_sha256=str(contract["contract_sha256"]),
    )


def read_frozen_stage0(
    frozen: FrozenStage0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for name, report in frozen.reports.items():
        path = {
            "fold_assignment": frozen.fold_assignment_path,
            "atlas_prototypes": frozen.atlas_path,
            "atlas_manifest": frozen.atlas_manifest_path,
            "trigger_ledger": frozen.trigger_path,
            "proposal_ledger": frozen.proposal_path,
            "input_manifest": frozen.input_manifest_path,
            "parent_control_manifest": frozen.parent_control_manifest_path,
        }[name]
        if artifact_report(path)["content_sha256"] != report["content_sha256"]:
            raise RuntimeError(f"Frozen {name} no longer matches its content SHA")
    trigger = pd.read_csv(
        frozen.trigger_path,
        dtype={
            "id": str,
            "well_id": str,
            "eligible": bool,
            "accepted_trigger": bool,
            "raw_gr_observed": bool,
        },
    )
    proposal = pd.read_csv(
        frozen.proposal_path,
        dtype={"event_id": str, "well_id": str},
    )
    input_manifest = pd.read_csv(
        frozen.input_manifest_path,
        dtype={"well_id": str},
    )
    if trigger["id"].duplicated().any():
        raise RuntimeError("Frozen trigger ledger contains duplicate ids")
    if not proposal.empty and proposal.duplicated(
        ["event_id", "proposal_rank"]
    ).any():
        raise RuntimeError("Frozen proposal ledger contains duplicate event ranks")
    return trigger, proposal, input_manifest


# %% [markdown]
# ## 9. Late truth and hidden-like attachment

# %%
def forward_window_mse(squared_error: np.ndarray, horizon: int) -> np.ndarray:
    values = np.asarray(squared_error, np.float64)
    if len(values) < int(horizon):
        return np.empty(0, dtype=np.float64)
    cumulative = np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])
    return (cumulative[horizon:] - cumulative[:-horizon]) / float(horizon)


def load_hidden_like_late(
    config: Mapping[str, Any],
    ledger: TruthAccessLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = get_nested(config, "data.hidden_like") or {}
    path = resolve_existing(
        str(spec["filename"]),
        [str(value) for value in spec.get("candidates", [])],
        [str(value) for value in spec.get("patterns", [])],
    )
    actual_sha = sha256_file(path)
    if actual_sha != str(spec["expected_sha256"]):
        raise ValueError("Hidden-like assignment SHA mismatch")
    role_columns = [str(value) for value in spec["role_columns"].values()]
    frame = pd.read_csv(
        path,
        usecols=["well_id", *role_columns],
        dtype={"well_id": str},
    )
    if frame["well_id"].duplicated().any():
        raise ValueError("Hidden-like assignment contains duplicate wells")
    ledger.record_hidden_late(len(frame))
    renamed = frame.rename(
        columns={
            role_column: scope
            for scope, role_column in spec["role_columns"].items()
        }
    )
    return renamed, {
        "path": str(path),
        "raw_sha256": actual_sha,
        "rows": int(len(frame)),
    }


def build_late_readouts(
    train_dir: Path,
    trigger_ledger: pd.DataFrame,
    proposal_ledger: pd.DataFrame,
    config: Mapping[str, Any],
    ledger: TruthAccessLedger,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not ledger.frozen:
        raise RuntimeError("Late readout requires completed pre-truth SHA freeze")
    horizon = int(get_nested(config, "validation.stage_0.bad_event_horizon_rows"))
    trigger_parts: list[pd.DataFrame] = []
    event_rows: list[dict[str, Any]] = []
    for well, well_rows in trigger_ledger.groupby("well_id", sort=True):
        ordered = well_rows.sort_values("suffix_offset", kind="mergesort").copy()
        horizontal_path = train_dir / f"{well}__horizontal_well.csv"
        truth_frame = pd.read_csv(horizontal_path, usecols=["TVT"])
        ledger.record_target_truth_late(len(truth_frame))
        truth = pd.to_numeric(truth_frame["TVT"], errors="coerce").to_numpy(np.float64)
        row_idx = ordered["row_idx"].to_numpy(np.int64)
        suffix_truth = truth[row_idx]
        base = ordered["saved_likpf_tvt"].to_numpy(np.float64)
        if not np.isfinite(suffix_truth).all():
            raise ValueError(f"well={well} suffix truth contains non-finite values")
        base_horizon_mse = forward_window_mse(np.square(base - suffix_truth), horizon)
        horizon_rmse = np.full(len(ordered), np.nan, dtype=np.float64)
        horizon_rmse[: len(base_horizon_mse)] = np.sqrt(base_horizon_mse)
        ordered["true_tvt"] = suffix_truth.astype(np.float32)
        ordered["base_abs_error"] = np.abs(base - suffix_truth).astype(np.float32)
        ordered["base_horizon_rmse"] = horizon_rmse.astype(np.float32)
        ordered["bad_event"] = horizon_rmse >= 10.0
        trigger_parts.append(ordered)

        accepted = ordered.loc[ordered["accepted_trigger"] & ordered["eligible"]]
        for trigger in accepted.itertuples(index=False):
            event_id = f"{well}_{int(trigger.row_idx)}"
            proposals = proposal_ledger.loc[
                proposal_ledger["event_id"].eq(event_id)
            ].sort_values("proposal_rank", kind="mergesort")
            target = float(trigger.true_tvt)
            proposal_values = pd.to_numeric(
                proposals["proposal_tvt"], errors="coerce"
            ).to_numpy(np.float64)
            proposal_errors = np.abs(proposal_values - target)
            atlas_within10 = bool(
                len(proposal_errors) and np.any(proposal_errors <= 10.0)
            )
            base_within10 = bool(float(trigger.base_abs_error) <= 10.0)
            event_rows.append(
                {
                    "event_id": event_id,
                    "well_id": well,
                    "fold": int(trigger.fold),
                    "trigger_row_idx": int(trigger.row_idx),
                    "trigger_suffix_offset": int(trigger.suffix_offset),
                    "md_since": float(trigger.md_since),
                    "true_tvt": target,
                    "base_likpf_tvt": float(trigger.saved_likpf_tvt),
                    "base_abs_error": float(trigger.base_abs_error),
                    "base_horizon_rmse": float(trigger.base_horizon_rmse),
                    "bad_event": bool(trigger.bad_event),
                    "proposal_count": int(len(proposals)),
                    "atlas_top3_within10": atlas_within10,
                    "base_likpf_within10": base_within10,
                    "coverage_delta": int(atlas_within10) - int(base_within10),
                    "best_atlas_abs_error": (
                        float(np.min(proposal_errors))
                        if len(proposal_errors)
                        else None
                    ),
                    "top1_atlas_abs_error": (
                        float(proposal_errors[0])
                        if len(proposal_errors)
                        else None
                    ),
                    "top1_zncc": (
                        float(proposals["zncc"].iloc[0])
                        if len(proposals)
                        else None
                    ),
                }
            )
    trigger_readout = (
        pd.concat(trigger_parts, ignore_index=True)
        .sort_values(["well_id", "suffix_offset"], kind="mergesort")
        .reset_index(drop=True)
    )
    event_readout = pd.DataFrame(event_rows)
    if event_readout.empty:
        event_readout = pd.DataFrame(
            columns=[
                "event_id",
                "well_id",
                "fold",
                "trigger_row_idx",
                "trigger_suffix_offset",
                "md_since",
                "true_tvt",
                "base_likpf_tvt",
                "base_abs_error",
                "base_horizon_rmse",
                "bad_event",
                "proposal_count",
                "atlas_top3_within10",
                "base_likpf_within10",
                "coverage_delta",
                "best_atlas_abs_error",
                "top1_atlas_abs_error",
                "top1_zncc",
            ]
        )
    hidden, hidden_report = load_hidden_like_late(config, ledger)
    trigger_readout = trigger_readout.merge(
        hidden,
        on="well_id",
        how="left",
        validate="many_to_one",
    )
    event_readout = event_readout.merge(
        hidden,
        on="well_id",
        how="left",
        validate="many_to_one",
    )
    return trigger_readout, event_readout, hidden_report


# %% [markdown]
# ## 10. Stage 0 metrics and promotion gates

# %%
def roc_auc_binary(labels: np.ndarray, scores: np.ndarray) -> float | None:
    label = np.asarray(labels, dtype=bool)
    score = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(score)
    label = label[valid]
    score = score[valid]
    positives = int(label.sum())
    negatives = int((~label).sum())
    if positives == 0 or negatives == 0:
        return None
    ranks = pd.Series(score).rank(method="average").to_numpy(np.float64)
    positive_rank_sum = float(ranks[label].sum())
    return float(
        (
            positive_rank_sum - positives * (positives + 1) / 2.0
        )
        / (positives * negatives)
    )


def trigger_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    eligible = frame.loc[frame["eligible"].astype(bool)]
    if eligible.empty:
        return {
            "eligible_rows": 0,
            "accepted_triggers": 0,
            "trigger_row_fraction": None,
            "bad_event_fraction": None,
            "trigger_bad_event_auc": None,
            "circular_bad_event_auc": None,
            "auc_gain_over_circular": None,
        }
    real_auc = roc_auc_binary(
        eligible["bad_event"].to_numpy(bool),
        eligible["trigger_score"].to_numpy(np.float64),
    )
    circular_auc = roc_auc_binary(
        eligible["bad_event"].to_numpy(bool),
        eligible["circular_trigger_score"].to_numpy(np.float64),
    )
    return {
        "eligible_rows": int(len(eligible)),
        "accepted_triggers": int(eligible["accepted_trigger"].sum()),
        "trigger_row_fraction": float(eligible["accepted_trigger"].mean()),
        "bad_event_fraction": float(eligible["bad_event"].mean()),
        "trigger_bad_event_auc": real_auc,
        "circular_bad_event_auc": circular_auc,
        "auc_gain_over_circular": (
            float(real_auc - circular_auc)
            if real_auc is not None and circular_auc is not None
            else None
        ),
    }


def event_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "events": 0,
            "event_wells": 0,
            "atlas_top3_within10_coverage": None,
            "base_likpf_within10_coverage": None,
            "coverage_gain_over_base_likpf": None,
            "mean_best_atlas_abs_error": None,
            "mean_base_abs_error": None,
            "mean_top1_zncc": None,
        }
    atlas_coverage = float(frame["atlas_top3_within10"].astype(bool).mean())
    base_coverage = float(frame["base_likpf_within10"].astype(bool).mean())
    return {
        "events": int(len(frame)),
        "event_wells": int(frame["well_id"].nunique()),
        "atlas_top3_within10_coverage": atlas_coverage,
        "base_likpf_within10_coverage": base_coverage,
        "coverage_gain_over_base_likpf": atlas_coverage - base_coverage,
        "mean_best_atlas_abs_error": float(
            pd.to_numeric(frame["best_atlas_abs_error"], errors="coerce").mean()
        ),
        "mean_base_abs_error": float(frame["base_abs_error"].mean()),
        "mean_top1_zncc": float(
            pd.to_numeric(frame["top1_zncc"], errors="coerce").mean()
        ),
    }


def scope_metrics(
    scope: str,
    trigger_frame: pd.DataFrame,
    event_frame: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "scope": scope,
        **trigger_metrics(trigger_frame),
        **event_metrics(event_frame),
    }


def build_scope_and_fold_metrics(
    trigger_readout: pd.DataFrame,
    event_readout: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scope_rows = [scope_metrics("overall", trigger_readout, event_readout)]
    for scope in ("hidden_like_spatial", "hidden_like_typewell_purged"):
        trigger_mask = trigger_readout[scope].astype(str).eq("valid")
        event_mask = (
            event_readout[scope].astype(str).eq("valid")
            if not event_readout.empty
            else np.zeros(0, dtype=bool)
        )
        scope_rows.append(
            scope_metrics(
                scope,
                trigger_readout.loc[trigger_mask],
                event_readout.loc[event_mask],
            )
        )
    fold_rows: list[dict[str, Any]] = []
    for fold in range(int(get_nested(config, "validation.n_folds"))):
        fold_rows.append(
            {
                **scope_metrics(
                    f"fold_{fold}",
                    trigger_readout.loc[trigger_readout["fold"].eq(fold)],
                    event_readout.loc[event_readout["fold"].eq(fold)],
                ),
                "fold": fold,
            }
        )
    return pd.DataFrame(scope_rows), pd.DataFrame(fold_rows)


def _at_least(value: Any, threshold: float) -> bool:
    return (
        value is not None
        and math.isfinite(float(value))
        and float(value) >= float(threshold)
    )


def _positive(value: Any) -> bool:
    return value is not None and math.isfinite(float(value)) and float(value) > 0.0


def scientific_row_checks(
    row: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, bool]:
    fraction_range = [float(value) for value in gates["trigger_row_fraction_range"]]
    fraction = row["trigger_row_fraction"]
    return {
        "trigger_bad_event_auc": _at_least(
            row["trigger_bad_event_auc"],
            float(gates["minimum_trigger_bad_event_auc"]),
        ),
        "auc_gain_over_circular": _at_least(
            row["auc_gain_over_circular"],
            float(gates["minimum_auc_gain_over_circular"]),
        ),
        "trigger_row_fraction": (
            fraction is not None
            and fraction_range[0] <= float(fraction) <= fraction_range[1]
        ),
        "atlas_top3_within10_coverage": _at_least(
            row["atlas_top3_within10_coverage"],
            float(gates["minimum_atlas_top3_within10_coverage"]),
        ),
        "coverage_gain_over_base_likpf": _at_least(
            row["coverage_gain_over_base_likpf"],
            float(gates["minimum_coverage_gain_over_base_likpf"]),
        ),
    }


def proposal_separation_is_valid(
    proposal_ledger: pd.DataFrame,
    top_k: int,
    minimum_separation_ft: float,
) -> bool:
    if proposal_ledger.empty:
        return True
    for _, rows in proposal_ledger.groupby("event_id", sort=False):
        ranks = rows["proposal_rank"].astype(int).tolist()
        if ranks != list(range(1, len(ranks) + 1)) or len(ranks) > int(top_k):
            return False
        values = rows["proposal_tvt"].to_numpy(np.float64)
        if len(values) > 1:
            distances = np.abs(values[:, None] - values[None, :])
            off_diagonal = distances[np.triu_indices(len(values), k=1)]
            if np.any(off_diagonal < float(minimum_separation_ft) - 1.0e-9):
                return False
    return True


def evaluate_stage0_gate(
    scope_frame: pd.DataFrame,
    fold_frame: pd.DataFrame,
    proposal_ledger: pd.DataFrame,
    input_manifest: pd.DataFrame,
    atlas_manifest: Mapping[str, Any],
    ledger: TruthAccessLedger,
    config: Mapping[str, Any],
    *,
    debug: bool,
) -> dict[str, Any]:
    stage0 = get_nested(config, "validation.stage_0") or {}
    atlas_spec = stage0.get("atlas") or {}
    gates = stage0.get("all_required") or {}
    overall = scope_frame.loc[scope_frame["scope"].eq("overall")].iloc[0].to_dict()
    overall_checks = scientific_row_checks(overall, gates)
    fold_checks: list[dict[str, Any]] = []
    for row in fold_frame.to_dict(orient="records"):
        checks = scientific_row_checks(row, gates)
        fold_checks.append(
            {
                "fold": int(row["fold"]),
                **checks,
                "passed": bool(all(checks.values())),
            }
        )
    passing_folds = sum(int(row["passed"]) for row in fold_checks)
    hidden_checks: dict[str, bool] = {}
    for scope in ("hidden_like_spatial", "hidden_like_typewell_purged"):
        row = scope_frame.loc[scope_frame["scope"].eq(scope)].iloc[0]
        hidden_checks[scope] = _positive(
            row["auc_gain_over_circular"]
        ) and _positive(row["coverage_gain_over_base_likpf"])
    atlas_fold_safe = all(
        int(row["validation_source_intersection_count"]) == 0
        for row in atlas_manifest["folds"]
    )
    expected_runs = int(stage0["diagnostic_pf_seed_well_runs"])
    actual_runs = int(input_manifest["pf_seed_well_runs"].sum())
    technical_checks = {
        "target_truth_rows_before_freeze_zero": (
            ledger.target_truth_rows_before_freeze == 0
        ),
        "hidden_role_rows_before_freeze_zero": (
            ledger.hidden_role_rows_before_freeze == 0
        ),
        "donor_fold_leakage_zero": ledger.donor_fold_leakage_violations == 0,
        "atlas_validation_source_intersection_zero": atlas_fold_safe,
        "proposal_topk_and_separation": proposal_separation_is_valid(
            proposal_ledger,
            int(atlas_spec["top_k"]),
            float(atlas_spec["minimum_tvt_separation_ft"]),
        ),
        "pf_seed_well_run_count": debug or actual_runs == expected_runs,
        "particles_500_seed_count_1": (
            int(stage0["particles"]) == 500
            and int(stage0["diagnostic_seed_count"]) == 1
        ),
        "no_stage1_or_inference": not any(
            bool(get_nested(config, key))
            for key in (
                "execution.run_stage_1",
                "execution.run_inference",
                "execution.create_submission",
            )
        ),
        "zero_lightgbm_and_boosters": (
            int(get_nested(config, "execution.stage_0_counts.lightgbm_configs")) == 0
            and int(get_nested(config, "execution.stage_0_counts.trained_folds")) == 0
            and int(get_nested(config, "execution.stage_0_counts.boosters")) == 0
        ),
        "zero_full_parent_pf_control_replays": int(
            get_nested(
                config,
                "execution.stage_0_counts.full_parent_pf_control_replays",
            )
        )
        == 0,
    }
    scientific_checks = {
        **overall_checks,
        "minimum_passing_folds": passing_folds
        >= int(gates["minimum_passing_folds"]),
        "positive_hidden_like_spatial_directions": hidden_checks[
            "hidden_like_spatial"
        ],
        "positive_hidden_like_typewell_purged_directions": hidden_checks[
            "hidden_like_typewell_purged"
        ],
    }
    technical_pass = bool(all(technical_checks.values()))
    scientific_pass = bool(all(scientific_checks.values()))
    stage1_eligible = technical_pass and scientific_pass and not debug
    return {
        "stage": "stage_0",
        "technical_gate_passed": technical_pass,
        "scientific_gate_passed": scientific_pass,
        "stage_1_eligible": stage1_eligible,
        "technical_checks": technical_checks,
        "scientific_checks": scientific_checks,
        "fold_checks": fold_checks,
        "passing_folds": passing_folds,
        "required_passing_folds": int(gates["minimum_passing_folds"]),
        "diagnostic_pf_seed_well_runs": actual_runs,
        "decision": (
            "debug_completed_no_promotion_decision"
            if debug
            else "stage0_pass_wait_for_separate_stage1_approval"
            if stage1_eligible
            else "stage0_failed_close_without_rejuvenation_pf"
        ),
    }


# %% [markdown]
# ## 11. Execution orchestration and generated artifacts

# %%
def run_stage0(
    config: Mapping[str, Any],
    *,
    maximum_wells: int | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    validate_scientific_contract(config, require_run_approval=True)
    require_authoritative_runtime()
    started = time.time()
    train_dir = resolve_train_dir(config)
    output_dir = output_directory(config)
    wells = discover_wells(train_dir, maximum_wells=maximum_wells)
    if not wells:
        raise RuntimeError("No paired raw train wells were found")
    saved_likpf, parent_control_report = load_saved_parent_likpf(
        config,
        maximum_wells=maximum_wells,
    )
    if sorted(saved_likpf["well_id"].unique().tolist()) != sorted(wells):
        raise ValueError("Raw train and saved exp072 likPF well identity mismatch")
    fold_by_well = deterministic_well_folds(
        wells,
        int(get_nested(config, "validation.n_folds")),
        int(get_nested(config, "validation.seed")),
    )
    fold_assignment = pd.DataFrame(
        [
            {"well_id": well, "fold": int(fold_by_well[well])}
            for well in sorted(wells)
        ]
    )
    ledger = TruthAccessLedger()
    contract = validate_scientific_contract(config)
    print("Experiment:", EXPERIMENT_NAME)
    print("Route:", get_nested(config, "experiment.route"))
    print("Parent:", get_nested(config, "lineage.parent"))
    print("Implementation scope:", get_nested(config, "execution.implementation_scope"))
    print("Raw train:", train_dir)
    print("Contract SHA256:", contract["contract_sha256"])
    print(
        "Saved exp072 likPF:",
        parent_control_report["path"],
        "| content_sha:",
        parent_control_report["content_sha256"],
    )
    print(
        f"Stage 0: 500 particles × 1 seed × {len(wells)} wells "
        f"= {len(wells)} diagnostic seed-well runs"
    )
    print("Stage 1: 500 particles × 128 seeds × 773 wells = 98,944 runs (disabled)")
    print("LightGBM configs / trained folds / boosters: 0 / 0 / 0")

    atlas_frame, atlas_manifest = build_fold_safe_atlases(
        wells,
        fold_by_well,
        train_dir,
        config,
        ledger,
    )
    trigger_parts: list[pd.DataFrame] = []
    proposal_parts: list[pd.DataFrame] = []
    manifest_rows: list[dict[str, Any]] = []
    for index, well in enumerate(sorted(wells), start=1):
        fold = int(fold_by_well[well])
        fold_atlas = atlas_frame.loc[atlas_frame["fold"].eq(fold)].reset_index(
            drop=True
        )
        trigger, proposal, manifest = build_prefreeze_rows_for_well(
            well,
            fold,
            saved_likpf.loc[saved_likpf["well_id"].eq(well)],
            train_dir,
            fold_atlas,
            config,
            ledger,
        )
        trigger_parts.append(trigger)
        if not proposal.empty:
            proposal_parts.append(proposal)
        manifest_rows.append(manifest)
        if index % 25 == 0 or index == len(wells):
            print(
                f"diagnostic PF [{index}/{len(wells)}] well={well} "
                f"events={manifest['accepted_trigger_rows']} "
                f"proposals={manifest['proposal_rows']}",
                flush=True,
            )
    trigger_ledger = (
        pd.concat(trigger_parts, ignore_index=True)
        .sort_values(["well_id", "suffix_offset"], kind="mergesort")
        .reset_index(drop=True)
    )
    proposal_ledger = (
        pd.concat(proposal_parts, ignore_index=True)
        .sort_values(["well_id", "trigger_suffix_offset", "proposal_rank"], kind="mergesort")
        .reset_index(drop=True)
        if proposal_parts
        else pd.DataFrame(
            columns=[
                "event_id",
                "well_id",
                "fold",
                "trigger_row_idx",
                "trigger_suffix_offset",
                "proposal_rank",
                "proposal_tvt",
                "zncc",
                "tvt_bin",
                "patch_count",
                "source_well_count",
                "source_wells_sha256",
            ]
        )
    )
    input_manifest = pd.DataFrame(manifest_rows).sort_values(
        "well_id", kind="mergesort"
    )
    raw_identity = input_manifest[
        ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"]
    ].reset_index(drop=True)
    raw_identity_sha = logical_dataframe_sha256(raw_identity)
    expected_raw_identity_sha = str(
        get_nested(config, "data.expected_raw_well_identity_sha256")
    )
    if not debug and raw_identity_sha != expected_raw_identity_sha:
        raise ValueError(
            "Raw train well-file identity mismatch: "
            f"expected={expected_raw_identity_sha} actual={raw_identity_sha}"
        )
    atlas_manifest["raw_well_identity_sha256"] = raw_identity_sha
    atlas_manifest["expected_raw_well_identity_sha256"] = expected_raw_identity_sha
    frozen = freeze_prefreeze_artifacts(
        output_dir,
        fold_assignment,
        atlas_frame,
        atlas_manifest,
        trigger_ledger,
        proposal_ledger,
        input_manifest,
        parent_control_report,
        contract,
        ledger,
    )
    frozen_trigger, frozen_proposal, frozen_input = read_frozen_stage0(frozen)
    trigger_readout, event_readout, hidden_report = build_late_readouts(
        train_dir,
        frozen_trigger,
        frozen_proposal,
        config,
        ledger,
    )
    scope_frame, fold_frame = build_scope_and_fold_metrics(
        trigger_readout,
        event_readout,
        config,
    )
    gate = evaluate_stage0_gate(
        scope_frame,
        fold_frame,
        frozen_proposal,
        frozen_input,
        atlas_manifest,
        ledger,
        config,
        debug=debug,
    )
    generated_paths = {
        "trigger_readout": output_dir / f"{OUTPUT_PREFIX}_trigger_readout.csv.gz",
        "event_readout": output_dir / f"{OUTPUT_PREFIX}_event_readout.csv.gz",
        "scope_metrics": output_dir / f"{OUTPUT_PREFIX}_scope_metrics.csv",
        "fold_metrics": output_dir / f"{OUTPUT_PREFIX}_fold_metrics.csv",
        "gate_report": output_dir / f"{OUTPUT_PREFIX}_gate_report.json",
        "summary": output_dir / f"{OUTPUT_PREFIX}_summary.json",
    }
    write_deterministic_csv_gzip(generated_paths["trigger_readout"], trigger_readout)
    write_deterministic_csv_gzip(generated_paths["event_readout"], event_readout)
    write_csv(generated_paths["scope_metrics"], scope_frame)
    write_csv(generated_paths["fold_metrics"], fold_frame)
    write_json(generated_paths["gate_report"], gate)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "stage": "stage_0",
        "decision": gate["decision"],
        "technical_gate_passed": gate["technical_gate_passed"],
        "scientific_gate_passed": gate["scientific_gate_passed"],
        "stage_1_eligible": gate["stage_1_eligible"],
        "created_at": datetime.now(UTC).isoformat(),
        "runtime_seconds": float(time.time() - started),
        "debug": bool(debug),
        "maximum_wells": maximum_wells,
        "wells": int(len(wells)),
        "rows": int(len(trigger_ledger)),
        "eligible_trigger_rows": int(trigger_ledger["eligible"].sum()),
        "accepted_triggers": int(trigger_ledger["accepted_trigger"].sum()),
        "proposal_rows": int(len(proposal_ledger)),
        "contract_sha256": contract["contract_sha256"],
        "truth_access": to_jsonable(ledger.__dict__),
        "hidden_like": hidden_report,
        "saved_parent_control": parent_control_report,
        "scope_metrics": scope_frame.to_dict(orient="records"),
        "fold_metrics": fold_frame.to_dict(orient="records"),
        "gate": gate,
        "execution_counts": {
            "scientific_variants": 0,
            "diagnostic_pf_replays": 1,
            "particles": 500,
            "seeds": 1,
            "seed_well_runs": int(frozen_input["pf_seed_well_runs"].sum()),
            "full_parent_pf_control_replays": 0,
            "lightgbm_configs": 0,
            "trained_folds": 0,
            "boosters": 0,
        },
        "runtime_versions": runtime_versions(),
        "prefreeze_artifacts": frozen.reports,
    }
    write_json(generated_paths["summary"], summary)
    summary["generated_artifacts"] = {
        key: artifact_report(path)
        for key, path in generated_paths.items()
        if key != "summary"
    }
    write_json(generated_paths["summary"], summary)

    metrics_path = (
        KAGGLE_WORKING_ROOT / "metrics.json"
        if is_kaggle_runtime()
        else locate_config().parent / "metrics.json"
    )
    write_json(
        metrics_path,
        {
            "experiment": EXPERIMENT_NAME,
            "route": "pf_beam",
            "status": gate["decision"],
            "updated_at": datetime.now(UTC).date().isoformat(),
            "cv": None,
            "public_lb": None,
            "private_lb": None,
            "metric": "trigger_auc_and_atlas_top3_coverage",
            "stage0": {
                "passed": gate["stage_1_eligible"],
                "gate": gate,
                "scope_metrics": scope_frame.to_dict(orient="records"),
                "fold_metrics": fold_frame.to_dict(orient="records"),
                "summary_path": str(generated_paths["summary"]),
                "summary_sha256": sha256_file(generated_paths["summary"]),
            },
            "notes": (
                "Stage 0 only. Stage 1 rejuvenation PF, inference, and "
                "submission remain disabled."
            ),
        },
    )
    log_summary = dict(summary)
    log_summary["summary_artifact"] = artifact_report(generated_paths["summary"])
    log_summary["metrics_artifact"] = artifact_report(metrics_path)
    print("EXP370_STAGE0_SUMMARY_BEGIN", flush=True)
    print(
        json.dumps(to_jsonable(log_summary), indent=2, sort_keys=True),
        flush=True,
    )
    print("EXP370_STAGE0_SUMMARY_END", flush=True)
    display(scope_frame)
    display(fold_frame)
    display(gate)
    return summary


# %% [markdown]
# ## 12. Setup and fail-closed execution selection

# %%
CONFIG = load_config()
CONTRACT = validate_scientific_contract(CONFIG)
SETUP = {
    "experiment": EXPERIMENT_NAME,
    "route": get_nested(CONFIG, "experiment.route"),
    "status": get_nested(CONFIG, "experiment.status"),
    "implementation_scope": get_nested(CONFIG, "execution.implementation_scope"),
    "contract_sha256": CONTRACT["contract_sha256"],
    "particles": CONTRACT["particles"],
    "diagnostic_seeds": CONTRACT["diagnostic_seed_count"],
    "planned_stage0_seed_well_runs": CONTRACT["diagnostic_pf_seed_well_runs"],
    "reporting_folds": get_nested(CONFIG, "validation.n_folds"),
    "run_stage_0": CONTRACT["run_stage_0"],
    "run_stage_1": CONTRACT["run_stage_1"],
    "run_inference": CONTRACT["run_inference"],
    "create_submission": CONTRACT["create_submission"],
}
display(pd.DataFrame([SETUP]))

# %%
if bool(get_nested(CONFIG, "execution.run_stage_0")) and get_ipython() is not None:
    DEBUG = os.environ.get("EXPERIMENT_DEBUG", "0") == "1"
    MAX_WELLS_TEXT = os.environ.get("EXPERIMENT_MAX_WELLS")
    MAX_WELLS = int(MAX_WELLS_TEXT) if MAX_WELLS_TEXT else None
    STAGE0_SUMMARY = run_stage0(
        CONFIG,
        maximum_wells=MAX_WELLS,
        debug=DEBUG,
    )
elif bool(get_nested(CONFIG, "execution.run_stage_0")):
    print(
        "exp370 Stage 0 is approved for the canonical Kaggle Notebook. "
        "Direct module import remains side-effect free."
    )
else:
    print(
        "exp370 Stage 0 implementation is ready, but execution.run_stage_0=false. "
        "Kaggle package/push/run, Stage 1, inference, and submission remain disabled."
    )
