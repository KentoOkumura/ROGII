# %% [markdown]
# # exp484 Student-t GR filtering likelihood-PF — Stage 0/1 train
#
# This compact self-contained candidate changes only the exp404 per-particle
# GR emission from capped Gaussian to the fixed exp374 Student-t score with
# `df=4`. Stage 0 is a stable-hash fixed32 technical preflight, not CV.
# Separately approved Stage 1 evaluates the same kernel once on all 773 train
# wells. Candidate predictions, diagnostics, and content SHA are frozen before
# saved controls, unknown-suffix TVT, folds, or hidden-like roles are read.

# %% [markdown]
# ## Contents
# 1. Imports and notebook contract
# 2. Notebook-safe configuration, path, and SHA helpers
# 3. Frozen scientific and execution contracts
# 4. Stable-hash fixed32 scope and truth-access ledger
# 5. Exact exp404 target-free input preparation
# 6. Student-t emission and likelihood-PF kernel
# 7. Candidate generation and prediction freeze
# 8. Saved-control and truth-late technical readout
# 9. Stage 0 technical gates and generated artifacts
# 10. All-well Stage 1 truth-late CV and promotion gate
# 11. Setup, configuration preview, and selected execution

# %% [markdown]
# ## 1. Imports and notebook contract

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
import resource
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ModuleNotFoundError:
    NUMBA_AVAILABLE = False

    def njit(*args: Any, **_: Any) -> Any:
        if args and callable(args[0]):
            return args[0]

        def decorator(function: Any) -> Any:
            return function

        return decorator


EXPERIMENT_NAME = "exp484_student_t_gr_filtering_likelihood_pf"
OUTPUT_PREFIX = EXPERIMENT_NAME
PRIMARY_CANDIDATE = "likpf_scale5_student_t_df4"
PRIMARY_CONTROL = "likpf_scale_5_x1p0"
SOURCE_FILENAME = f"{EXPERIMENT_NAME}_compact_selfcontained_train.py"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP484_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


# %% [markdown]
# ## 2. Notebook-safe configuration, path, and SHA helpers

# %%
def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        item = float(value)
        return item if math.isfinite(item) else None
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def mapping_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def get_nested(
    config: Mapping[str, Any],
    dotted_key: str,
    default: Any = None,
) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


def candidate_package_dirs() -> list[Path]:
    root = project_root()
    candidates = [
        Path.cwd(),
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


def load_experiment_config(package_dir: Path | None = None) -> dict[str, Any]:
    candidates = [package_dir] if package_dir is not None else candidate_package_dirs()
    checked: list[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        path = candidate / "config.yaml"
        checked.append(str(path))
        config = read_yaml(path)
        if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
            return config
    raise FileNotFoundError(f"exp484 config not found; checked={checked}")


def artifact_dir() -> Path:
    path = (
        KAGGLE_WORKING_ROOT / "artifacts"
        if KAGGLE_WORKING_ROOT.exists()
        else project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def metrics_output_path() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        return KAGGLE_WORKING_ROOT / "metrics.json"
    return project_root() / "experiments" / EXPERIMENT_NAME / "metrics.json"


def train_data_dir(config: Mapping[str, Any]) -> Path:
    if KAGGLE_INPUT_ROOT.exists():
        fixed = (
            KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction" / "train",
            KAGGLE_INPUT_ROOT
            / "competitions"
            / "rogii-wellbore-geology-prediction"
            / "train",
        )
        for candidate in fixed:
            if next(candidate.glob("*__horizontal_well.csv"), None) is not None:
                return candidate
        for candidate in sorted(KAGGLE_INPUT_ROOT.glob("**/train")):
            if next(candidate.glob("*__horizontal_well.csv"), None) is not None:
                return candidate
    return project_root() / str(get_nested(config, "data.train_dir", "data/raw/train"))


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_decompressed_csv(path: str | Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_content_sha(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> str:
    digest = hashlib.sha256()
    for column in columns:
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


def dataframe_schema_sha(frame: pd.DataFrame) -> str:
    schema = [(str(column), str(frame[column].dtype)) for column in frame.columns]
    return mapping_sha256(schema)


def write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")
    return {"path": str(path), "raw_sha256": sha256_path(path)}


def write_deterministic_gzip_csv(
    frame: pd.DataFrame,
    path: Path,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            frame.to_csv(zipped, index=False, lineterminator="\n")
    return {
        "path": str(path),
        "rows": int(len(frame)),
        "columns": frame.columns.astype(str).tolist(),
        "schema_sha256": dataframe_schema_sha(frame),
        "raw_sha256": sha256_path(path),
        "decompressed_sha256": sha256_decompressed_csv(path),
    }


def stable_seed(*parts: object, modulo: int = 2_147_483_647) -> int:
    key = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo + 1


def peak_rss_gb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0**2 if platform.system() != "Darwin" else 1024.0**3
    return value / divisor


def runtime_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyyaml": getattr(yaml, "__version__", "unknown"),
        "numba_available": NUMBA_AVAILABLE,
    }
    if NUMBA_AVAILABLE:
        import numba

        versions["numba"] = numba.__version__
    return versions


def resolve_existing(
    filename: str,
    candidates: Iterable[str],
    patterns: Iterable[str] = (),
) -> Path:
    checked: list[str] = []
    for raw in candidates:
        candidate = Path(str(raw))
        direct = candidate if candidate.name == filename else candidate / filename
        checked.append(str(direct))
        if direct.exists():
            return direct
        if candidate.exists() and candidate.is_dir():
            for pattern in patterns:
                matches = sorted(candidate.glob(str(pattern)))
                if len(matches) == 1:
                    return matches[0]
                if len(matches) > 1:
                    raise ValueError(f"ambiguous {filename}: {matches}")
    local_matches = sorted(project_root().glob(f"**/{filename}"))
    if len(local_matches) == 1:
        return local_matches[0]
    checked.extend(str(path) for path in local_matches)
    raise FileNotFoundError(f"could not resolve {filename}; checked={checked}")


def resolve_bootstrap_asset(filename: str, local_path: str) -> Path:
    candidates = [
        Path.cwd() / "assets" / filename,
        project_root() / local_path,
        KAGGLE_WORKING_ROOT / "assets" / filename,
    ]
    if KAGGLE_INPUT_ROOT.exists():
        candidates.extend(sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}")))
    matches = [candidate for candidate in candidates if candidate.exists()]
    if not matches:
        raise FileNotFoundError(f"bootstrap asset not found: {filename}")
    return matches[0]


# %% [markdown]
# ## 3. Frozen scientific and execution contracts
#
# The state-independent Student-t normalization constant is intentionally
# omitted. The inherited exp404 `1e-300` numerical likelihood floor remains,
# but there is no z-score, squared-residual, or log-emission clipping.

# %%
def student_t_log_emission(
    zscore: np.ndarray | float,
    *,
    df: float = 4.0,
) -> np.ndarray:
    values = np.asarray(zscore, dtype=np.float64)
    return -0.5 * (float(df) + 1.0) * np.log1p((values * values) / float(df))


def student_t_formula_contract(df: float = 4.0) -> dict[str, Any]:
    if float(df) != 4.0:
        raise ValueError("exp484 fixes Student-t df=4")
    z = np.asarray([0.0, 0.25, 1.0, 4.0, 16.0, 1.0e6], dtype=np.float64)
    observed = student_t_log_emission(z, df=df)
    expected = -0.5 * (df + 1.0) * np.log1p((z * z) / df)
    center = 1.0e-5
    center_observed = float(student_t_log_emission(center, df=df))
    center_quadratic = -0.5 * (df + 1.0) / df * center * center
    likelihood = np.exp(observed)
    normalized = likelihood / likelihood.sum()
    checks = {
        "fixed_df4": bool(float(df) == 4.0),
        "formula_exact": bool(np.array_equal(observed, expected)),
        "zero_log_score": bool(observed[0] == 0.0),
        "symmetric": bool(
            np.array_equal(
                student_t_log_emission(z, df=df),
                student_t_log_emission(-z, df=df),
            )
        ),
        "monotone_in_abs_z": bool(np.all(np.diff(observed) < 0.0)),
        "gaussian_center_quadratic": bool(
            abs(center_observed - center_quadratic) <= 1.0e-18
        ),
        "extreme_score_finite": bool(np.isfinite(observed).all()),
        "finite_positive_weights": bool(
            np.isfinite(likelihood).all()
            and np.all(likelihood > 0.0)
            and np.isfinite(normalized).all()
            and abs(float(normalized.sum()) - 1.0) <= 1.0e-15
        ),
    }
    return {
        "df": float(df),
        "formula": "-0.5*(df+1)*log1p(z^2/df)",
        "normalization_constant": "omitted_state_independent",
        "additional_clip": None,
        "center_quadratic_coefficient": -0.5 * (df + 1.0) / df,
        "center_absolute_error": abs(center_observed - center_quadratic),
        "minimum_extreme_likelihood": float(likelihood.min()),
        "checks": checks,
        "passed": bool(all(checks.values())),
    }


def validate_execution_contract(
    config: Mapping[str, Any],
    *,
    require_run_approval: bool,
) -> dict[str, int]:
    expected = {
        "active_variants": 1,
        "stage_0_candidate_pf_well_runs": 32,
        "stage_0_seed_well_trajectories": 4096,
        "stage_0_particle_starts": 2048000,
        "stage_1_candidate_pf_well_runs": 773,
        "stage_1_seed_well_trajectories": 98944,
        "stage_1_particle_starts": 49472000,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "control_pf_well_runs": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
    }
    observed = {
        key: int(get_nested(config, f"execution.{key}", -1)) for key in expected
    }
    if observed != expected:
        raise ValueError(f"exp484 execution count contract changed: {observed}")
    run_stage0 = bool(get_nested(config, "execution.run_stage_0", False))
    run_stage1 = bool(get_nested(config, "execution.run_stage_1", False))
    terminal_stage1 = bool(
        get_nested(config, "experiment.status")
        == "stage1_gate_failed_terminal_close"
        and get_nested(config, "execution.stage_1_kernel_status") == "COMPLETE"
        and get_nested(config, "stage_1_result.status")
        == "stage1_gate_failed_terminal_close"
    )
    if run_stage0 and run_stage1:
        raise ValueError("exp484 permits exactly one active execution stage")
    if run_stage1 and not bool(
        get_nested(config, "stage_0_result.all_technical_gates_passed", False)
    ):
        raise RuntimeError("exp484 Stage 1 requires a recorded Stage 0 technical PASS")
    if require_run_approval:
        if not bool(get_nested(config, "execution.kaggle_push_approved", False)):
            raise RuntimeError("exp484 Kaggle push is not approved")
        if run_stage0 and not bool(
            get_nested(config, "execution.stage_0_execution_approved", False)
        ):
            raise RuntimeError("exp484 Stage 0 execution is not approved")
        if run_stage1 and not bool(
            get_nested(config, "execution.stage_1_execution_approved", False)
        ):
            raise RuntimeError("exp484 Stage 1 execution is not approved")
        if not (run_stage0 or run_stage1 or terminal_stage1):
            raise RuntimeError("exp484 has no approved execution stage selected")
    return observed


def build_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    fixed = dict(get_nested(config, "model.fixed_from_exp404") or {})
    factor = dict(get_nested(config, "model.changed_factor") or {})
    contract: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "scientific_parent": get_nested(config, "lineage.parent"),
        "implementation_reference": get_nested(
            config,
            "lineage.exact_pf_implementation_reference",
        ),
        "hmm_evidence_source": get_nested(config, "lineage.hmm_evidence_source"),
        "primary_candidate": PRIMARY_CANDIDATE,
        "primary_control": PRIMARY_CONTROL,
        "changed_factor": {
            "family": str(factor["family"]),
            "df": float(factor["df"]),
            "formula": str(factor["log_likelihood"]),
            "normalization_constant": str(factor["normalization_constant"]),
            "additional_clip": factor["additional_clip"],
            "application_scope": str(factor["application_scope"]),
        },
        "fixed_from_exp404": fixed,
        "seed_policy": {
            "namespace": ["likpf", "train", "well_id"],
            "variant_name_in_seed": False,
            "seed_indices": [0, int(fixed["seeds"]) - 1],
        },
        "stage0": dict(get_nested(config, "stages.stage_0") or {}),
        "stage1": dict(get_nested(config, "stages.stage_1") or {}),
        "saved_control_rerun": False,
        "truth_attachment": get_nested(config, "validation.truth_attachment"),
        "formula_contract": student_t_formula_contract(float(factor["df"])),
        "forbidden": get_nested(config, "guards.forbidden"),
    }
    contract["scientific_contract_sha256"] = mapping_sha256(contract)
    return contract


def validate_scientific_contract(
    config: Mapping[str, Any],
    *,
    require_run_approval: bool = False,
) -> dict[str, Any]:
    expected: dict[str, Any] = {
        "experiment.name": EXPERIMENT_NAME,
        "experiment.route": "pf_beam",
        "experiment.status": "stage1_gate_failed_terminal_close",
        "lineage.parent": "exp417_scale5_seed_aggregation_promotion_audit",
        "lineage.exact_pf_implementation_reference": (
            "exp404_scale5_sigma_gr_likelihood_pf_ablation"
        ),
        "implementation.enabled": True,
        "implementation.implementation_approval_received": True,
        "implementation.canonical_train_notebook_adopted": True,
        "implementation.canonical_inference_notebook_adopted": False,
        "implementation.inference_enabled": False,
        "implementation.submission_enabled": False,
        "model.active_variants": ["student_t_df4"],
        "model.changed_factor.family": "student_t",
        "model.changed_factor.df": 4.0,
        "model.changed_factor.normalization_constant": (
            "omitted_as_particle_state_independent"
        ),
        "model.changed_factor.additional_clip": None,
        "model.fixed_from_exp404.particles": 500,
        "model.fixed_from_exp404.seeds": 128,
        "model.fixed_from_exp404.primary_seed_weighting_temperature": 5.0,
        "model.fixed_from_exp404.gr_scale_multiplier": 1.0,
        "model.fixed_from_exp404.momentum": 0.998,
        "model.fixed_from_exp404.rate_noise": 0.002,
        "model.fixed_from_exp404.position_noise": 0.005,
        "model.fixed_from_exp404.rough_position": 0.1,
        "model.fixed_from_exp404.rough_rate": 0.001,
        "model.fixed_from_exp404.resample_threshold_fraction": 0.5,
        "model.fixed_from_exp404.typewell_grid_step_ft": 0.2,
        "model.fixed_from_exp404.typewell_tvt_pad_ft": 100.0,
        "model.fixed_from_exp404.output_dtype": "float32",
        "stages.stage_0.well_set": "stable_sha256_fixed32",
        "stages.stage_0.scientific_variants": 1,
        "stages.stage_0.candidate_pf_well_runs": 32,
        "stages.stage_1.scientific_variants": 1,
        "stages.stage_1.candidate_pf_well_runs": 773,
        "execution.run_stage_1": False,
        "execution.run_inference": False,
        "execution.create_submission": False,
    }
    for key, required in expected.items():
        observed = get_nested(config, key)
        if observed != required:
            raise ValueError(
                f"exp484 scientific contract mismatch: {key}={observed!r}, "
                f"expected={required!r}"
            )
    base_clip = [
        float(value)
        for value in get_nested(config, "model.fixed_from_exp404.base_scale_clip")
    ]
    if base_clip != [10.0, 60.0]:
        raise ValueError("exp484 fixes the exp404 base GR scale clip to [10, 60]")
    formula = str(get_nested(config, "model.changed_factor.log_likelihood"))
    if formula != "-0.5*(df+1)*log1p(z^2/df)":
        raise ValueError("exp484 Student-t formula changed")
    validate_execution_contract(
        config,
        require_run_approval=require_run_approval,
    )
    return build_scientific_contract(config)


# %% [markdown]
# ## 4. Stable-hash fixed32 scope and truth-access ledger
#
# Fixed32 membership is selected only from well IDs by
# `sha256("exp484::stage0::<well_id>")`. The checked-in manifest contains no
# suffix truth, error, fold, hidden-like role, or result-derived field.

# %%
@dataclass
class TruthAccessLedger:
    prediction_frozen: bool = False
    truth_rows_before_freeze: int = 0
    control_rows_before_freeze: int = 0
    fold_rows_before_freeze: int = 0
    hidden_like_rows_before_freeze: int = 0
    truth_rows_after_freeze: int = 0
    control_rows_after_freeze: int = 0
    fold_rows_after_freeze: int = 0
    hidden_like_rows_after_freeze: int = 0

    def _record(self, label: str, rows: int) -> None:
        location = "after_freeze" if self.prediction_frozen else "before_freeze"
        field_name = f"{label}_rows_{location}"
        setattr(self, field_name, int(getattr(self, field_name)) + int(rows))
        if not self.prediction_frozen:
            raise RuntimeError(f"{label} was read before candidate prediction freeze")

    def record_truth(self, rows: int) -> None:
        self._record("truth", rows)

    def record_control(self, rows: int) -> None:
        self._record("control", rows)

    def record_fold(self, rows: int) -> None:
        self._record("fold", rows)

    def record_hidden_like(self, rows: int) -> None:
        self._record("hidden_like", rows)

    def mark_frozen(self) -> None:
        before = self.report()["before_freeze"]
        if any(int(value) != 0 for value in before.values()):
            raise RuntimeError("truth/reporting values were accessed before prediction freeze")
        self.prediction_frozen = True

    def require_frozen(self) -> None:
        if not self.prediction_frozen:
            raise RuntimeError("late readout requires a frozen candidate prediction")

    def report(self) -> dict[str, Any]:
        return {
            "prediction_frozen": self.prediction_frozen,
            "before_freeze": {
                "truth_rows": self.truth_rows_before_freeze,
                "control_rows": self.control_rows_before_freeze,
                "fold_rows": self.fold_rows_before_freeze,
                "hidden_like_rows": self.hidden_like_rows_before_freeze,
            },
            "after_freeze": {
                "truth_rows": self.truth_rows_after_freeze,
                "control_rows": self.control_rows_after_freeze,
                "fold_rows": self.fold_rows_after_freeze,
                "hidden_like_rows": self.hidden_like_rows_after_freeze,
            },
        }


def validate_raw_well_identity(
    config: Mapping[str, Any],
    raw_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizontal_path in sorted(raw_dir.glob("*__horizontal_well.csv")):
        well = horizontal_path.name.replace("__horizontal_well.csv", "")
        typewell_path = raw_dir / f"{well}__typewell.csv"
        if not typewell_path.exists():
            raise FileNotFoundError(typewell_path)
        rows.append(
            {
                "well_id": well,
                "horizontal_raw_sha256": sha256_path(horizontal_path),
                "typewell_raw_sha256": sha256_path(typewell_path),
            }
        )
    frame = (
        pd.DataFrame(rows)
        .sort_values("well_id", kind="mergesort")
        .reset_index(drop=True)
    )
    logical_sha = dataframe_content_sha(
        frame,
        ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"],
    )
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_sha = str(get_nested(config, "data.expected_raw_well_identity_sha256"))
    if len(frame) != expected_wells or logical_sha != expected_sha:
        raise ValueError("exp484 raw train well identity mismatch")
    return frame, {
        "path": str(raw_dir),
        "wells": len(frame),
        "logical_sha256": logical_sha,
    }


def fixed32_manifest_path(config: Mapping[str, Any]) -> Path:
    spec = dict(get_nested(config, "data.fixed32_manifest") or {})
    return resolve_bootstrap_asset(str(spec["filename"]), str(spec["local"]))


def load_fixed32_manifest(
    config: Mapping[str, Any],
    raw_identity: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = dict(get_nested(config, "data.fixed32_manifest") or {})
    path = fixed32_manifest_path(config)
    observed_raw_sha = sha256_path(path)
    if observed_raw_sha != str(spec["expected_sha256"]):
        raise ValueError("exp484 fixed32 manifest SHA mismatch")
    manifest = pd.read_csv(
        path,
        dtype={"well_id": str, "selection_sha256": str},
    )
    required = ["well_id", "selection_sha256", "total_rows", "suffix_rows"]
    if list(manifest.columns) != required:
        raise ValueError("exp484 fixed32 manifest schema changed")
    manifest["total_rows"] = pd.to_numeric(
        manifest["total_rows"],
        errors="raise",
    ).astype(np.int64)
    manifest["suffix_rows"] = pd.to_numeric(
        manifest["suffix_rows"],
        errors="raise",
    ).astype(np.int64)
    namespace = str(spec["selection_namespace"])
    candidates = raw_identity[["well_id"]].copy()
    candidates["selection_sha256"] = candidates["well_id"].map(
        lambda well: hashlib.sha256(f"{namespace}::{well}".encode("utf-8")).hexdigest()
    )
    expected = (
        candidates.sort_values(
            ["selection_sha256", "well_id"],
            kind="mergesort",
        )
        .head(int(spec["expected_wells"]))
        .reset_index(drop=True)
    )
    if not manifest[["well_id", "selection_sha256"]].equals(expected):
        raise ValueError("exp484 fixed32 is not the stable-hash first32")
    if manifest["well_id"].duplicated().any():
        raise ValueError("exp484 fixed32 contains duplicate wells")
    if int(manifest["suffix_rows"].sum()) != int(spec["expected_suffix_rows"]):
        raise ValueError("exp484 fixed32 suffix row count changed")
    return manifest, {
        "path": str(path),
        "raw_sha256": observed_raw_sha,
        "wells": len(manifest),
        "suffix_rows": int(manifest["suffix_rows"].sum()),
        "selection_namespace": namespace,
        "logical_sha256": dataframe_content_sha(manifest, required),
    }


def selected_raw_input_manifest(
    raw_dir: Path,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in manifest.itertuples(index=False):
        well = str(row.well_id)
        horizontal_path = raw_dir / f"{well}__horizontal_well.csv"
        typewell_path = raw_dir / f"{well}__typewell.csv"
        target_free = pd.read_csv(horizontal_path, usecols=["TVT_input"])
        total_rows = len(target_free)
        suffix_rows = int(target_free["TVT_input"].isna().sum())
        if total_rows != int(row.total_rows) or suffix_rows != int(row.suffix_rows):
            raise ValueError(f"{well}: fixed32 target-free row counts changed")
        rows.append(
            {
                "well_id": well,
                "selection_sha256": str(row.selection_sha256),
                "total_rows": total_rows,
                "suffix_rows": suffix_rows,
                "horizontal_raw_sha256": sha256_path(horizontal_path),
                "typewell_raw_sha256": sha256_path(typewell_path),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("well_id", kind="mergesort")
        .reset_index(drop=True)
    )


# %% [markdown]
# ## 5. Exact exp404 target-free input preparation

# %%
def load_horizontal_without_truth(well: str, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / f"{well}__horizontal_well.csv"
    frame = pd.read_csv(path, usecols=["MD", "Z", "GR", "TVT_input"])
    frame = frame[["MD", "Z", "GR", "TVT_input"]]
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["MD", "Z"]].isna().any().any():
        raise ValueError(f"{well}: MD/Z must be finite")
    return frame


def load_typewell(well: str, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / f"{well}__typewell.csv"
    frame = pd.read_csv(path, usecols=["TVT", "GR"])
    frame["TVT"] = pd.to_numeric(frame["TVT"], errors="coerce")
    frame["GR"] = pd.to_numeric(frame["GR"], errors="coerce")
    frame = (
        frame.dropna(subset=["TVT"])
        .sort_values("TVT", kind="mergesort")
        .reset_index(drop=True)
    )
    if len(frame) < 2 or not np.isfinite(frame["TVT"].to_numpy(np.float64)).all():
        raise ValueError(f"{well}: Type Well TVT support is invalid")
    typewell_mean = float(frame["GR"].mean())
    if not math.isfinite(typewell_mean):
        raise ValueError(f"{well}: Type Well GR mean is not finite")
    frame["GR"] = frame["GR"].fillna(typewell_mean)
    return frame


def uniform_typewell_grid(
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    *,
    step: float = 0.2,
) -> tuple[np.ndarray, float, float]:
    minimum = float(np.min(typewell_tvt))
    maximum = float(np.max(typewell_tvt))
    grid_tvt = np.arange(minimum, maximum + step, step)
    grid_gr = np.interp(grid_tvt, typewell_tvt, typewell_gr).astype(np.float64)
    return grid_gr, minimum, float(step)


def exp404_base_gr_scale(
    horizontal: pd.DataFrame,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    *,
    clip: tuple[float, float] = (10.0, 60.0),
) -> dict[str, Any]:
    known = horizontal["TVT_input"].notna().to_numpy()
    if not known.any():
        raise ValueError("likelihood-PF requires at least one known-prefix row")
    known_tvt = horizontal.loc[known, "TVT_input"].to_numpy(np.float64)
    known_gr = horizontal.loc[known, "GR"].fillna(0.0).to_numpy(np.float64)
    typewell_at_known = np.interp(known_tvt, typewell_tvt, typewell_gr)
    residual = known_gr - typewell_at_known
    raw_scale = float(np.nanstd(residual))
    if not math.isfinite(raw_scale):
        raise ValueError("known-prefix GR residual scale is not finite")
    return {
        "raw_scale": raw_scale,
        "base_scale": float(np.clip(raw_scale, clip[0], clip[1])),
        "known_rows": int(known.sum()),
        "known_gr_missing_rows": int(horizontal.loc[known, "GR"].isna().sum()),
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual, ddof=0)),
        "base_clip_min": float(clip[0]),
        "base_clip_max": float(clip[1]),
    }


def exp404_initial_rate(
    horizontal: pd.DataFrame,
    *,
    tail_rows: int = 30,
) -> float:
    known = horizontal.loc[horizontal["TVT_input"].notna()].tail(tail_rows)
    delta_tvt = np.diff(known["TVT_input"].to_numpy(np.float64))
    delta_z = np.diff(known["Z"].to_numpy(np.float64))
    delta_md = np.diff(known["MD"].to_numpy(np.float64))
    valid = (
        (delta_md > 0.0)
        & np.isfinite(delta_md)
        & np.isfinite(delta_tvt)
        & np.isfinite(delta_z)
    )
    if int(valid.sum()) < 3:
        return 0.0
    return float(
        np.median((delta_tvt[valid] + delta_z[valid]) / delta_md[valid])
    )


def prepare_likelihood_pf_inputs(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    grid_step: float,
) -> dict[str, Any]:
    typewell_tvt = typewell["TVT"].to_numpy(np.float64)
    typewell_gr = typewell["GR"].to_numpy(np.float64)
    known_mask = horizontal["TVT_input"].notna().to_numpy()
    eval_mask = ~known_mask
    if not known_mask.any() or not eval_mask.any():
        raise ValueError("likelihood-PF requires non-empty known prefix and suffix")
    known = horizontal.loc[known_mask]
    evaluation = horizontal.loc[eval_mask]
    last_known = known.iloc[-1]
    last_known_tvt = float(last_known["TVT_input"])
    last_known_md = float(last_known["MD"])
    last_position = last_known_tvt + float(last_known["Z"])
    scale_audit = exp404_base_gr_scale(
        horizontal,
        typewell_tvt,
        typewell_gr,
    )
    grid_gr, grid_minimum, actual_step = uniform_typewell_grid(
        typewell_tvt,
        typewell_gr,
        step=grid_step,
    )
    typewell_mean = float(typewell_gr.mean())
    interpolated_gr = (
        horizontal["GR"]
        .interpolate(limit_direction="both")
        .fillna(typewell_mean)
        .to_numpy(np.float64)
    )
    eval_indices = np.flatnonzero(eval_mask).astype(np.int64)
    eval_md = evaluation["MD"].to_numpy(np.float64)
    eval_gr = interpolated_gr[eval_indices]
    if not np.isfinite(eval_gr).all():
        raise ValueError("evaluation GR interpolation is not finite")
    return {
        "eval_indices": eval_indices,
        "eval_md": eval_md,
        "eval_z": evaluation["Z"].to_numpy(np.float64),
        "eval_gr": eval_gr,
        "raw_gr_observed": evaluation["GR"].notna().to_numpy(bool),
        "md_since": eval_md - last_known_md,
        "last_known_tvt": last_known_tvt,
        "last_known_position": last_position,
        "initial_rate": exp404_initial_rate(horizontal),
        "grid_gr": grid_gr,
        "grid_minimum": grid_minimum,
        "grid_step": actual_step,
        "scale_audit": {
            **scale_audit,
            "candidate_scale": float(scale_audit["base_scale"]),
            "multiplier": 1.0,
            "post_multiplier_clip_applied": False,
            "post_multiplier_clip_count": 0,
        },
    }


# %% [markdown]
# ## 6. Student-t emission and likelihood-PF kernel
#
# Dynamics, initialization, position clamp, ESS threshold, systematic
# resampling, roughening, seed order, and temperature-5 aggregation are copied
# from exp404. The only scientific line change is the particle likelihood.

# %%
@njit(cache=True)
def _interp1(
    grid: np.ndarray,
    value: float,
    minimum: float,
    step: float,
) -> float:
    index = int((value - minimum) / step)
    if index < 0:
        return grid[0]
    final = len(grid) - 1
    if index >= final:
        return grid[final]
    fraction = (value - minimum) / step - index
    return grid[index] * (1.0 - fraction) + grid[index + 1] * fraction


@njit(cache=True, nogil=True)
def _pf_student_t_allseeds(
    md_v: np.ndarray,
    z_v: np.ndarray,
    gr_v: np.ndarray,
    grid_gr: np.ndarray,
    grid_minimum: float,
    grid_step: float,
    gr_scale: float,
    student_df: float,
    last_position: float,
    initial_rate: float,
    particles: int,
    seeds: int,
    seed_base: int,
    momentum: float,
    rate_noise: float,
    position_noise: float,
    rough_position: float,
    rough_rate: float,
    resample_fraction: float,
    initial_spread: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exact exp404 PF with only the fixed df=4 Student-t GR score changed."""
    rows = len(md_v)
    predictions = np.empty((seeds, rows))
    log_likelihoods = np.empty(seeds)
    resampling_counts = np.zeros(seeds, np.int64)
    minimum_ess = np.full(seeds, float(particles))
    position_clip_counts = np.zeros(seeds, np.int64)
    grid_maximum = grid_minimum + len(grid_gr) * grid_step
    for seed_index in range(seeds):
        np.random.seed(seed_base + seed_index)
        position = np.empty(particles)
        rate = np.empty(particles)
        weights = np.ones(particles) / particles
        for particle in range(particles):
            position[particle] = (
                last_position + initial_spread * np.random.randn()
            )
            rate[particle] = initial_rate + 0.01 * np.random.randn()
        log_likelihood = 0.0
        previous_md = md_v[0] - 1.0
        for row in range(rows):
            delta_md = md_v[row] - previous_md
            if delta_md < 1.0:
                delta_md = 1.0
            for particle in range(particles):
                rate[particle] = (
                    momentum * rate[particle] + rate_noise * np.random.randn()
                )
                position[particle] += (
                    rate[particle] * delta_md
                    + position_noise * np.random.randn()
                )
                tvt_value = position[particle] - z_v[row]
                if tvt_value < grid_minimum - 100.0:
                    tvt_value = grid_minimum - 100.0
                    position_clip_counts[seed_index] += 1
                if tvt_value > grid_maximum + 100.0:
                    tvt_value = grid_maximum + 100.0
                    position_clip_counts[seed_index] += 1
                position[particle] = tvt_value + z_v[row]
            average_likelihood = 0.0
            for particle in range(particles):
                expected_gr = _interp1(
                    grid_gr,
                    position[particle] - z_v[row],
                    grid_minimum,
                    grid_step,
                )
                zscore = (gr_v[row] - expected_gr) / gr_scale
                log_emission = (
                    -0.5
                    * (student_df + 1.0)
                    * np.log1p((zscore * zscore) / student_df)
                )
                likelihood = np.exp(log_emission)
                if likelihood < 1e-300:
                    likelihood = 1e-300
                average_likelihood += weights[particle] * likelihood
                weights[particle] *= likelihood
            if average_likelihood < 1e-300:
                average_likelihood = 1e-300
            log_likelihood += np.log(average_likelihood)
            weight_sum = 0.0
            for particle in range(particles):
                weight_sum += weights[particle]
            if weight_sum > 0.0:
                for particle in range(particles):
                    weights[particle] /= weight_sum
            else:
                for particle in range(particles):
                    weights[particle] = 1.0 / particles
            inverse_ess = 0.0
            for particle in range(particles):
                inverse_ess += weights[particle] * weights[particle]
            effective_sample_size = 1.0 / inverse_ess
            if effective_sample_size < minimum_ess[seed_index]:
                minimum_ess[seed_index] = effective_sample_size
            if effective_sample_size < resample_fraction * particles:
                cumulative = np.empty(particles)
                cumulative_value = 0.0
                for particle in range(particles):
                    cumulative_value += weights[particle]
                    cumulative[particle] = cumulative_value
                initial_uniform = np.random.uniform(0.0, 1.0 / particles)
                new_position = np.empty(particles)
                new_rate = np.empty(particles)
                cursor = 0
                for particle in range(particles):
                    uniform = initial_uniform + particle / particles
                    while (
                        cursor < particles - 1
                        and cumulative[cursor] < uniform
                    ):
                        cursor += 1
                    new_position[particle] = (
                        position[cursor] + rough_position * np.random.randn()
                    )
                    new_rate[particle] = (
                        rate[cursor] + rough_rate * np.random.randn()
                    )
                for particle in range(particles):
                    position[particle] = new_position[particle]
                    rate[particle] = new_rate[particle]
                    weights[particle] = 1.0 / particles
                resampling_counts[seed_index] += 1
            estimate = 0.0
            for particle in range(particles):
                estimate += weights[particle] * (
                    position[particle] - z_v[row]
                )
            predictions[seed_index, row] = estimate
            previous_md = md_v[row]
        log_likelihoods[seed_index] = log_likelihood
    return (
        predictions,
        log_likelihoods,
        resampling_counts,
        minimum_ess,
        position_clip_counts,
    )


def aggregate_temperature5(
    predictions: np.ndarray,
    log_likelihoods: np.ndarray,
    *,
    temperature: float = 5.0,
) -> np.ndarray:
    if float(temperature) != 5.0:
        raise ValueError("exp484 fixes seed aggregation temperature to 5")
    centered = log_likelihoods - float(np.max(log_likelihoods))
    weights = np.exp(centered / float(temperature))
    weights /= weights.sum()
    if not np.isfinite(weights).all() or not np.isclose(
        weights.sum(),
        1.0,
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise ValueError("Student-t seed aggregation weights are invalid")
    return (weights[:, None] * predictions).sum(axis=0)


def run_student_t_pf(
    prepared: Mapping[str, Any],
    *,
    particles: int,
    seeds: int,
    seed_base: int,
    student_df: float,
    temperature: float,
    momentum: float,
    rate_noise: float,
    position_noise: float,
    rough_position: float,
    rough_rate: float,
    resample_fraction: float,
    initial_spread: float,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray]:
    started = time.time()
    (
        predictions,
        log_likelihoods,
        resampling_counts,
        minimum_ess,
        position_clip_counts,
    ) = _pf_student_t_allseeds(
        np.asarray(prepared["eval_md"], dtype=np.float64),
        np.asarray(prepared["eval_z"], dtype=np.float64),
        np.asarray(prepared["eval_gr"], dtype=np.float64),
        np.asarray(prepared["grid_gr"], dtype=np.float64),
        float(prepared["grid_minimum"]),
        float(prepared["grid_step"]),
        float(prepared["scale_audit"]["candidate_scale"]),
        float(student_df),
        float(prepared["last_known_position"]),
        float(prepared["initial_rate"]),
        int(particles),
        int(seeds),
        int(seed_base),
        float(momentum),
        float(rate_noise),
        float(position_noise),
        float(rough_position),
        float(rough_rate),
        float(resample_fraction),
        float(initial_spread),
    )
    output = aggregate_temperature5(
        predictions,
        log_likelihoods,
        temperature=temperature,
    )
    diagnostics = {
        "runtime_seconds": time.time() - started,
        "seed_loglik_mean_per_row": (
            float(log_likelihoods.mean()) / len(prepared["eval_md"])
        ),
        "seed_loglik_best_per_row": (
            float(log_likelihoods.max()) / len(prepared["eval_md"])
        ),
        "seed_loglik_spread": float(log_likelihoods.std()),
        "resampling_count_total": int(resampling_counts.sum()),
        "resampling_count_min": int(resampling_counts.min()),
        "resampling_count_max": int(resampling_counts.max()),
        "minimum_ess_min": float(minimum_ess.min()),
        "minimum_ess_mean": float(minimum_ess.mean()),
        "minimum_ess_max": float(minimum_ess.max()),
        "position_clip_count_total": int(position_clip_counts.sum()),
        "seed_prediction_std_mean": float(predictions.std(axis=0).mean()),
        "prediction_finite_fraction": float(np.isfinite(output).mean()),
        "log_likelihood_finite_fraction": float(
            np.isfinite(log_likelihoods).mean()
        ),
    }
    return output, diagnostics, predictions, log_likelihoods


def warm_up_pf_kernel() -> None:
    _pf_student_t_allseeds(
        np.linspace(1.0, 8.0, 8),
        np.zeros(8),
        np.full(8, 50.0),
        np.linspace(45.0, 55.0, 100),
        0.0,
        0.2,
        20.0,
        4.0,
        50.0,
        0.0,
        8,
        2,
        1,
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        4.5,
    )


# %% [markdown]
# ## 7. Candidate generation and prediction freeze

# %%
def decode_target_free_well(
    well: str,
    raw_dir: Path,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.time()
    horizontal = load_horizontal_without_truth(well, raw_dir)
    typewell = load_typewell(well, raw_dir)
    fixed = dict(get_nested(config, "model.fixed_from_exp404") or {})
    factor = dict(get_nested(config, "model.changed_factor") or {})
    prepared = prepare_likelihood_pf_inputs(
        horizontal,
        typewell,
        grid_step=float(fixed["typewell_grid_step_ft"]),
    )
    seed_base = stable_seed("likpf", "train", well)
    output, diagnostics, _, _ = run_student_t_pf(
        prepared,
        particles=int(fixed["particles"]),
        seeds=int(fixed["seeds"]),
        seed_base=seed_base,
        student_df=float(factor["df"]),
        temperature=float(fixed["primary_seed_weighting_temperature"]),
        momentum=float(fixed["momentum"]),
        rate_noise=float(fixed["rate_noise"]),
        position_noise=float(fixed["position_noise"]),
        rough_position=float(fixed["rough_position"]),
        rough_rate=float(fixed["rough_rate"]),
        resample_fraction=float(fixed["resample_threshold_fraction"]),
        initial_spread=float(fixed["initial_position_spread_ft"]),
    )
    eval_indices = np.asarray(prepared["eval_indices"], dtype=np.int64)
    raw_observed = np.asarray(prepared["raw_gr_observed"], dtype=bool)
    missing_fraction = float(1.0 - raw_observed.mean())
    candidate = pd.DataFrame(
        {
            "id": [f"{well}_{int(row)}" for row in eval_indices],
            "well_id": str(well),
            "row_idx": eval_indices,
            "suffix_offset": np.arange(len(eval_indices), dtype=np.int64),
            "last_known_tvt": np.float64(prepared["last_known_tvt"]),
            "md_since": np.asarray(prepared["md_since"], dtype=np.float64),
            "raw_gr_observed": raw_observed,
            "well_missing_fraction": np.float64(missing_fraction),
            PRIMARY_CANDIDATE: output.astype(np.float32),
        }
    )
    scale = dict(prepared["scale_audit"])
    audit = {
        "well_id": str(well),
        "status": "ok",
        "prefix_rows": int(scale["known_rows"]),
        "prefix_gr_missing_rows": int(scale["known_gr_missing_rows"]),
        "eval_rows": len(candidate),
        "eval_raw_gr_observed_rows": int(raw_observed.sum()),
        "eval_raw_gr_missing_rows": int((~raw_observed).sum()),
        "eval_raw_gr_missing_fraction": missing_fraction,
        "last_known_tvt": float(prepared["last_known_tvt"]),
        "last_known_position": float(prepared["last_known_position"]),
        "initial_rate": float(prepared["initial_rate"]),
        "gr_scale_raw": float(scale["raw_scale"]),
        "gr_scale_base": float(scale["base_scale"]),
        "gr_scale_candidate": float(scale["candidate_scale"]),
        "gr_scale_multiplier": float(scale["multiplier"]),
        "student_t_df": float(factor["df"]),
        "additional_emission_clip": None,
        "seed_base": int(seed_base),
        "seed_first": int(seed_base),
        "seed_last": int(seed_base + int(fixed["seeds"]) - 1),
        "seeds": int(fixed["seeds"]),
        "particles": int(fixed["particles"]),
        "pf_well_runs": 1,
        "seed_well_trajectories": int(fixed["seeds"]),
        "particle_starts": int(fixed["seeds"]) * int(fixed["particles"]),
        **diagnostics,
        "wall_seconds": time.time() - started,
    }
    if not np.isfinite(candidate[PRIMARY_CANDIDATE].to_numpy(np.float64)).all():
        raise ValueError(f"{well}: Student-t PF prediction is non-finite")
    return candidate, audit


def freeze_target_free_predictions(
    raw_dir: Path,
    output: Path,
    config: Mapping[str, Any],
    manifest: pd.DataFrame,
    ledger: TruthAccessLedger,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    if ledger.prediction_frozen:
        raise RuntimeError("exp484 prediction ledger is already frozen")
    wells = manifest["well_id"].astype(str).tolist()
    expected = int(get_nested(config, "stages.stage_0.candidate_pf_well_runs"))
    if len(wells) != expected or len(set(wells)) != expected:
        raise ValueError("exp484 Stage 0 requires exactly 32 unique wells")
    warm_up_pf_kernel()
    results = Parallel(
        n_jobs=int(get_nested(config, "runtime.num_workers")),
        prefer="threads",
    )(
        delayed(decode_target_free_well)(well, raw_dir, config)
        for well in wells
    )
    prediction = pd.concat([item[0] for item in results], ignore_index=True)
    audit = pd.DataFrame([item[1] for item in results])
    prediction = (
        prediction.sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    audit = (
        audit.sort_values("well_id", kind="mergesort")
        .reset_index(drop=True)
    )
    if prediction.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("exp484 candidate row identity is duplicated")
    expected_rows = int(get_nested(config, "data.fixed32_manifest.expected_suffix_rows"))
    if (
        len(prediction) != expected_rows
        or prediction["well_id"].nunique() != expected
        or len(audit) != expected
        or not audit["status"].eq("ok").all()
    ):
        raise ValueError("exp484 fixed32 candidate coverage mismatch")
    prediction_path = output / f"{OUTPUT_PREFIX}_stage0_predictions.csv.gz"
    audit_path = output / f"{OUTPUT_PREFIX}_stage0_well_audit.csv"
    prediction_artifact = write_deterministic_gzip_csv(
        prediction,
        prediction_path,
    )
    audit.to_csv(audit_path, index=False)
    logical_columns = ["id", "well_id", "row_idx", PRIMARY_CANDIDATE]
    frozen = {
        "frozen_before_truth_or_control_attachment": True,
        "rows": len(prediction),
        "wells": int(prediction["well_id"].nunique()),
        "prediction_columns": [PRIMARY_CANDIDATE],
        "logical_columns": logical_columns,
        "logical_content_sha256": dataframe_content_sha(
            prediction,
            logical_columns,
        ),
        "schema_sha256": dataframe_schema_sha(prediction),
        "raw_gzip_sha256": prediction_artifact["raw_sha256"],
        "decompressed_content_sha256": prediction_artifact[
            "decompressed_sha256"
        ],
        "well_audit_sha256": sha256_path(audit_path),
        "truth_access_ledger_before_freeze": ledger.report(),
    }
    ledger.mark_frozen()
    return prediction, audit, frozen, {
        "prediction": prediction_artifact,
        "well_audit": {
            "path": str(audit_path),
            "raw_sha256": sha256_path(audit_path),
        },
    }


def require_frozen_prediction(frozen: Mapping[str, Any]) -> None:
    if not bool(frozen.get("frozen_before_truth_or_control_attachment")):
        raise RuntimeError("late attachment requires frozen exp484 prediction")
    if len(str(frozen.get("logical_content_sha256") or "")) != 64:
        raise RuntimeError("frozen exp484 logical prediction SHA is missing")


# %% [markdown]
# ## 8. Saved-control and truth-late technical readout
#
# The saved exp404 file is raw/decompressed-SHA checked only after candidate
# freeze. Its pre-serialization logical SHA is retained as provenance, while
# the aligned fixed32 subset receives a fresh logical SHA. Fixed32 RMSE is a
# diagnostic, not CV and not a Stage 1 promotion decision.

# %%
def restore_frozen_float32_column(
    values: pd.Series,
    *,
    label: str,
) -> pd.Series:
    restored = pd.to_numeric(values, errors="raise").to_numpy(dtype=np.float32)
    if not np.isfinite(restored).all():
        raise ValueError(f"{label} contains non-finite values")
    return pd.Series(restored, index=values.index, name=values.name)


def saved_control_path(config: Mapping[str, Any]) -> Path:
    spec = dict(get_nested(config, "data.saved_control") or {})
    return resolve_existing(
        str(spec["filename"]),
        [str(value) for value in spec.get("candidates", [])],
        [str(value) for value in spec.get("patterns", [])],
    )


def load_saved_control_after_freeze(
    prediction: pd.DataFrame,
    frozen: Mapping[str, Any],
    config: Mapping[str, Any],
    ledger: TruthAccessLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    require_frozen_prediction(frozen)
    ledger.require_frozen()
    if dataframe_content_sha(
        prediction,
        list(frozen["logical_columns"]),
    ) != str(frozen["logical_content_sha256"]):
        raise ValueError("exp484 candidate changed after freeze")
    spec = dict(get_nested(config, "data.saved_control") or {})
    path = saved_control_path(config)
    raw_sha = sha256_path(path)
    decompressed_sha = sha256_decompressed_csv(path)
    if raw_sha != str(spec["expected_raw_sha256"]):
        raise ValueError("exp484 saved exp404 control raw SHA mismatch")
    if decompressed_sha != str(spec["expected_decompressed_sha256"]):
        raise ValueError("exp484 saved exp404 control decompressed SHA mismatch")
    source = pd.read_csv(
        path,
        compression="gzip",
        usecols=["id", PRIMARY_CONTROL],
        dtype={"id": str},
    )
    source[PRIMARY_CONTROL] = restore_frozen_float32_column(
        source[PRIMARY_CONTROL],
        label=f"saved exp404 {PRIMARY_CONTROL}",
    )
    ledger.record_control(len(source))
    if source["id"].duplicated().any():
        raise ValueError("saved exp404 control contains duplicate IDs")
    aligned = source.set_index("id").reindex(prediction["id"].astype(str))
    if aligned[PRIMARY_CONTROL].isna().any():
        raise ValueError("saved exp404 control is incomplete for fixed32")
    control = pd.DataFrame(
        {
            "id": prediction["id"].astype(str).to_numpy(),
            PRIMARY_CONTROL: aligned[PRIMARY_CONTROL].to_numpy(np.float32),
        }
    )
    subset_logical_sha = dataframe_content_sha(
        control,
        ["id", PRIMARY_CONTROL],
    )
    return control, {
        "path": str(path),
        "raw_sha256": raw_sha,
        "decompressed_sha256": decompressed_sha,
        "source_prediction_logical_sha256": str(
            spec["expected_logical_sha256"]
        ),
        "source_logical_sha_policy": (
            "record_frozen_pre_serialization_provenance; raw and decompressed "
            "artifact SHA are the executable input guards"
        ),
        "subset_logical_sha256": subset_logical_sha,
        "rows_read_after_freeze": len(source),
        "subset_rows": len(control),
        "prediction_column": PRIMARY_CONTROL,
        "control_pf_well_reruns": 0,
    }


def load_suffix_truth_after_freeze(
    wells: Sequence[str],
    raw_dir: Path,
    ledger: TruthAccessLedger,
) -> pd.DataFrame:
    ledger.require_frozen()
    parts: list[pd.DataFrame] = []
    for well in wells:
        horizontal = pd.read_csv(
            raw_dir / f"{well}__horizontal_well.csv",
            usecols=["TVT_input", "TVT"],
        )
        tvt_input = pd.to_numeric(horizontal["TVT_input"], errors="coerce")
        truth = pd.to_numeric(horizontal["TVT"], errors="coerce")
        eval_indices = np.flatnonzero(tvt_input.isna().to_numpy()).astype(np.int64)
        values = truth.iloc[eval_indices].to_numpy(np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{well}: suffix truth is non-finite")
        parts.append(
            pd.DataFrame(
                {
                    "id": [f"{well}_{int(row)}" for row in eval_indices],
                    "true_tvt": values,
                }
            )
        )
    frame = pd.concat(parts, ignore_index=True)
    ledger.record_truth(len(frame))
    return frame


def attach_truth_late_readout(
    prediction: pd.DataFrame,
    frozen: Mapping[str, Any],
    config: Mapping[str, Any],
    raw_dir: Path,
    ledger: TruthAccessLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    control, control_report = load_saved_control_after_freeze(
        prediction,
        frozen,
        config,
        ledger,
    )
    wells = sorted(prediction["well_id"].astype(str).unique().tolist())
    truth = load_suffix_truth_after_freeze(wells, raw_dir, ledger)
    frame = prediction.merge(
        control,
        on="id",
        how="left",
        validate="one_to_one",
    ).merge(
        truth,
        on="id",
        how="left",
        validate="one_to_one",
    )
    finite_columns = [PRIMARY_CANDIDATE, PRIMARY_CONTROL, "true_tvt"]
    if not np.isfinite(frame[finite_columns].to_numpy(np.float64)).all():
        raise ValueError("exp484 truth-late readout is non-finite")
    return frame, {
        "truth_attached_after_prediction_freeze": True,
        "saved_control_attached_after_prediction_freeze": True,
        "candidate_content_sha256_reverified": dataframe_content_sha(
            prediction,
            list(frozen["logical_columns"]),
        ),
        "saved_control": control_report,
        "truth_access_ledger": ledger.report(),
    }


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def fixed32_diagnostic_metrics(
    frame: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    truth = frame["true_tvt"].to_numpy(np.float64)
    candidate = frame[PRIMARY_CANDIDATE].to_numpy(np.float64)
    control = frame[PRIMARY_CONTROL].to_numpy(np.float64)
    candidate_rmse = rmse(truth, candidate)
    control_rmse = rmse(truth, control)
    rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well_id", sort=True):
        well_truth = group["true_tvt"].to_numpy(np.float64)
        well_candidate = group[PRIMARY_CANDIDATE].to_numpy(np.float64)
        well_control = group[PRIMARY_CONTROL].to_numpy(np.float64)
        candidate_value = rmse(well_truth, well_candidate)
        control_value = rmse(well_truth, well_control)
        rows.append(
            {
                "well_id": str(well),
                "rows": len(group),
                "candidate_rmse": candidate_value,
                "control_rmse": control_value,
                "delta_candidate_minus_control": candidate_value - control_value,
            }
        )
    by_well = pd.DataFrame(rows)
    return {
        "scope": "stable_hash_fixed32_not_cv",
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "candidate_rmse": candidate_rmse,
        "control_rmse": control_rmse,
        "gain_control_minus_candidate_ft": control_rmse - candidate_rmse,
        "improved_wells": int(
            by_well["delta_candidate_minus_control"].lt(0.0).sum()
        ),
        "by_well_delta_p95": float(
            by_well["delta_candidate_minus_control"].quantile(0.95)
        ),
        "worst_well_delta": float(
            by_well["delta_candidate_minus_control"].max()
        ),
        "promotion_evidence": False,
    }, by_well


# %% [markdown]
# ## 9. Stage 0 technical gates and generated artifacts

# %%
def evaluate_stage0_technical_gate(
    config: Mapping[str, Any],
    manifest: pd.DataFrame,
    prediction: pd.DataFrame,
    audit: pd.DataFrame,
    frozen: Mapping[str, Any],
    raw_report: Mapping[str, Any],
    manifest_report: Mapping[str, Any],
    late_attachment: Mapping[str, Any],
    ledger: TruthAccessLedger,
    *,
    candidate_seconds: float,
    rss_gb: float,
) -> dict[str, Any]:
    fixed = dict(get_nested(config, "model.fixed_from_exp404") or {})
    expected = dict(get_nested(config, "stages.stage_0") or {})
    formula = student_t_formula_contract(
        float(get_nested(config, "model.changed_factor.df"))
    )
    actual_counts = {
        "scientific_variants": 1,
        "candidate_pf_well_runs": int(audit["pf_well_runs"].sum()),
        "seed_well_trajectories": int(
            audit["seed_well_trajectories"].sum()
        ),
        "particle_starts": int(audit["particle_starts"].sum()),
        "control_pf_well_runs": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
    }
    expected_counts = {
        "scientific_variants": int(expected["scientific_variants"]),
        "candidate_pf_well_runs": int(expected["candidate_pf_well_runs"]),
        "seed_well_trajectories": int(expected["seed_well_trajectories"]),
        "particle_starts": int(expected["particle_starts"]),
        "control_pf_well_runs": int(expected["control_pf_well_runs"]),
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
    }
    seed_identity = bool(
        all(
            int(seed) == stable_seed("likpf", "train", str(well))
            for well, seed in zip(
                audit["well_id"],
                audit["seed_base"],
                strict=True,
            )
        )
    )
    finite_coverage = float(
        np.isfinite(prediction[PRIMARY_CANDIDATE].to_numpy(np.float64)).mean()
    )
    projected_full_seconds = candidate_seconds / len(manifest) * 773.0
    before_freeze = ledger.report()["before_freeze"]
    control_report = dict(late_attachment["saved_control"])
    measurements = {
        "prediction_rows": len(prediction),
        "prediction_wells": int(prediction["well_id"].nunique()),
        "audit_wells": len(audit),
        "finite_prediction_coverage": finite_coverage,
        "seed_identity_stable": seed_identity,
        "gr_scale_min": float(audit["gr_scale_candidate"].min()),
        "gr_scale_max": float(audit["gr_scale_candidate"].max()),
        "student_t_df_values": sorted(
            audit["student_t_df"].astype(float).unique().tolist()
        ),
        "minimum_ess_min": float(audit["minimum_ess_min"].min()),
        "minimum_ess_max": float(audit["minimum_ess_max"].max()),
        "resampling_count_total": int(
            audit["resampling_count_total"].sum()
        ),
        "resampling_count_min": int(audit["resampling_count_min"].min()),
        "resampling_count_max": int(audit["resampling_count_max"].max()),
        "position_clip_count_total": int(
            audit["position_clip_count_total"].sum()
        ),
        "candidate_seconds": candidate_seconds,
        "projected_full_seconds": projected_full_seconds,
        "peak_rss_gb": rss_gb,
        "execution_counts": actual_counts,
        "expected_execution_counts": expected_counts,
        "prediction_logical_sha256": frozen["logical_content_sha256"],
        "prediction_schema_sha256": frozen["schema_sha256"],
        "well_audit_sha256": frozen["well_audit_sha256"],
        "raw_identity_sha256": raw_report["logical_sha256"],
        "fixed32_manifest_sha256": manifest_report["raw_sha256"],
        "truth_access_before_freeze": before_freeze,
        "saved_control": control_report,
    }
    checks = {
        "formula_unit_contract": bool(formula["passed"]),
        "fixed32_manifest_contract": bool(
            len(manifest) == 32
            and int(manifest["suffix_rows"].sum())
            == int(get_nested(config, "data.fixed32_manifest.expected_suffix_rows"))
        ),
        "raw_identity_contract": bool(
            raw_report["logical_sha256"]
            == str(get_nested(config, "data.expected_raw_well_identity_sha256"))
        ),
        "all_wells_completed": bool(
            len(audit) == 32 and audit["status"].eq("ok").all()
        ),
        "finite_prediction_coverage": bool(finite_coverage == 1.0),
        "stable_seed_identity": seed_identity,
        "fixed_df4": bool(
            measurements["student_t_df_values"] == [4.0]
        ),
        "fixed_x1p0_gr_scale": bool(
            measurements["gr_scale_min"] >= 10.0
            and measurements["gr_scale_max"] <= 60.0
            and np.array_equal(
                audit["gr_scale_candidate"].to_numpy(np.float64),
                audit["gr_scale_base"].to_numpy(np.float64),
            )
        ),
        "ess_ledger_valid": bool(
            measurements["minimum_ess_min"] > 0.0
            and measurements["minimum_ess_max"] <= float(fixed["particles"])
        ),
        "resampling_ledger_valid": bool(
            measurements["resampling_count_total"] >= 0
            and measurements["resampling_count_min"] >= 0
        ),
        "truth_late": bool(
            all(int(value) == 0 for value in before_freeze.values())
            and bool(late_attachment["truth_attached_after_prediction_freeze"])
            and bool(
                late_attachment[
                    "saved_control_attached_after_prediction_freeze"
                ]
            )
        ),
        "prediction_and_audit_sha_present": bool(
            len(str(frozen["logical_content_sha256"])) == 64
            and len(str(frozen["schema_sha256"])) == 64
            and len(str(frozen["well_audit_sha256"])) == 64
        ),
        "saved_control_sha_contract": bool(
            control_report["raw_sha256"]
            == str(get_nested(config, "data.saved_control.expected_raw_sha256"))
            and control_report["decompressed_sha256"]
            == str(
                get_nested(
                    config,
                    "data.saved_control.expected_decompressed_sha256",
                )
            )
        ),
        "execution_count_match": bool(actual_counts == expected_counts),
        "runtime_projection_within_limit": bool(
            projected_full_seconds
            <= float(get_nested(config, "guards.technical.maximum_seconds_full_projection"))
        ),
        "peak_rss_within_limit": bool(
            rss_gb
            <= float(get_nested(config, "guards.technical.maximum_peak_rss_gb"))
        ),
    }
    return {
        "stage": "stage0_stable_hash_fixed32_technical_preflight_not_cv",
        "checks": checks,
        "passed": bool(all(checks.values())),
        "stage1_eligible_pending_separate_user_approval": bool(
            all(checks.values())
        ),
        "formula_contract": formula,
        "measurements": measurements,
        "failure_action": (
            "close_without_df_scale_temperature_clip_mixture_particle_seed_"
            "gate_blend_selector_or_same_oof_rescue"
        ),
    }


def require_kaggle_runtime() -> None:
    if KAGGLE_WORKING_ROOT.exists():
        return
    if os.environ.get("EXPERIMENT_ALLOW_LOCAL") == "1":
        return
    raise RuntimeError("exp484 authoritative Stage 0 execution is Kaggle CPU only")


def run_stage0(config: Mapping[str, Any]) -> dict[str, Any]:
    scientific_contract = validate_scientific_contract(
        config,
        require_run_approval=True,
    )
    require_kaggle_runtime()
    started = time.time()
    output = artifact_dir()
    raw_dir = train_data_dir(config)
    raw_identity, raw_report = validate_raw_well_identity(config, raw_dir)
    manifest, manifest_report = load_fixed32_manifest(config, raw_identity)
    selected_inputs = selected_raw_input_manifest(raw_dir, manifest)
    selected_input_path = output / f"{OUTPUT_PREFIX}_stage0_input_manifest.csv"
    selected_inputs.to_csv(selected_input_path, index=False)
    scientific_contract_artifact = write_json(
        output / f"{OUTPUT_PREFIX}_scientific_contract.json",
        scientific_contract,
    )
    ledger = TruthAccessLedger()
    candidate_started = time.time()
    prediction, audit, frozen, generated = freeze_target_free_predictions(
        raw_dir,
        output,
        config,
        manifest,
        ledger,
    )
    candidate_seconds = time.time() - candidate_started
    late_frame, late_attachment = attach_truth_late_readout(
        prediction,
        frozen,
        config,
        raw_dir,
        ledger,
    )
    diagnostic, by_well = fixed32_diagnostic_metrics(late_frame)
    truth_late_artifact = write_deterministic_gzip_csv(
        late_frame,
        output / f"{OUTPUT_PREFIX}_stage0_truth_late_readout.csv.gz",
    )
    by_well_path = output / f"{OUTPUT_PREFIX}_stage0_by_well.csv"
    by_well.to_csv(by_well_path, index=False)
    rss_gb = peak_rss_gb()
    gate = evaluate_stage0_technical_gate(
        config,
        manifest,
        prediction,
        audit,
        frozen,
        raw_report,
        manifest_report,
        late_attachment,
        ledger,
        candidate_seconds=candidate_seconds,
        rss_gb=rss_gb,
    )
    gate_artifact = write_json(
        output / f"{OUTPUT_PREFIX}_stage0_technical_gate.json",
        gate,
    )
    status = (
        "stage0_passed_pending_separate_stage1_approval"
        if gate["passed"]
        else "stage0_failed_closed"
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": status,
        "stage": "stage0_stable_hash_fixed32_technical_preflight_not_cv",
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "scientific_contract_sha256": scientific_contract[
            "scientific_contract_sha256"
        ],
        "counts": {
            "scientific_variants": 1,
            "candidate_pf_well_runs": int(audit["pf_well_runs"].sum()),
            "seed_well_trajectories": int(
                audit["seed_well_trajectories"].sum()
            ),
            "particle_starts": int(audit["particle_starts"].sum()),
            "saved_control_pf_well_runs": 0,
            "lightgbm_configs": 0,
            "trained_folds": 0,
            "boosters": 0,
            "hmm_well_runs": 0,
            "beam_well_runs": 0,
            "gpu_runs": 0,
        },
        "frozen_prediction": frozen,
        "late_attachment": late_attachment,
        "fixed32_diagnostic_not_cv": diagnostic,
        "technical_gate": gate,
        "runtime": {
            "candidate_seconds": candidate_seconds,
            "total_seconds": time.time() - started,
            "projected_full_seconds": gate["measurements"][
                "projected_full_seconds"
            ],
            "peak_rss_gb": rss_gb,
            "versions": runtime_versions(),
        },
        "artifacts": {
            **generated,
            "scientific_contract": scientific_contract_artifact,
            "selected_input_manifest": {
                "path": str(selected_input_path),
                "raw_sha256": sha256_path(selected_input_path),
                "logical_sha256": dataframe_content_sha(
                    selected_inputs,
                    selected_inputs.columns.astype(str).tolist(),
                ),
            },
            "fixed32_manifest": manifest_report,
            "truth_late_readout": truth_late_artifact,
            "by_well": {
                "path": str(by_well_path),
                "raw_sha256": sha256_path(by_well_path),
            },
            "technical_gate": gate_artifact,
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "inference": False,
        "submission": False,
        "next_action": (
            "request_separate_stage1_approval"
            if gate["passed"]
            else "close_branch_without_rescue"
        ),
    }
    summary_artifact = write_json(
        output / f"{OUTPUT_PREFIX}_stage0_summary.json",
        summary,
    )
    summary["artifacts"]["summary"] = summary_artifact
    write_json(metrics_output_path(), summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 10. All-well Stage 1 truth-late CV and promotion gate
#
# The separately approved Stage 1 runs the unchanged Student-t PF once for
# every train well. File existence is checked before generation, but saved
# prediction/control contents, suffix truth, reporting folds, and hidden-like
# roles are parsed only after the all-well candidate and its SHA are frozen.

# %%
def freeze_stage1_target_free_predictions(
    results: Sequence[tuple[pd.DataFrame, dict[str, Any]]],
    output: Path,
    config: Mapping[str, Any],
    ledger: TruthAccessLedger,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    if ledger.prediction_frozen:
        raise RuntimeError("exp484 Stage 1 prediction ledger is already frozen")
    ordered = sorted(results, key=lambda item: str(item[1]["well_id"]))
    prediction = (
        pd.concat([item[0] for item in ordered], ignore_index=True)
        .sort_values(["well_id", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )
    audit = (
        pd.DataFrame([item[1] for item in ordered])
        .sort_values("well_id", kind="mergesort")
        .reset_index(drop=True)
    )
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if (
        len(prediction) != expected_rows
        or prediction["well_id"].nunique() != expected_wells
        or len(audit) != expected_wells
        or not audit["status"].eq("ok").all()
        or prediction["id"].duplicated().any()
        or prediction.duplicated(["well_id", "row_idx"]).any()
    ):
        raise ValueError("exp484 Stage 1 prediction identity or coverage changed")
    prediction_path = output / f"{OUTPUT_PREFIX}_stage1_predictions.csv.gz"
    audit_path = output / f"{OUTPUT_PREFIX}_stage1_well_audit.csv"
    prediction_artifact = write_deterministic_gzip_csv(
        prediction,
        prediction_path,
    )
    audit.to_csv(audit_path, index=False)
    logical_columns = ["id", "well_id", "row_idx", PRIMARY_CANDIDATE]
    logical_sha = dataframe_content_sha(prediction, logical_columns)
    readback = pd.read_csv(
        prediction_path,
        compression="gzip",
        dtype={
            "id": str,
            "well_id": str,
            PRIMARY_CANDIDATE: np.float32,
        },
    )
    readback_sha = dataframe_content_sha(readback, logical_columns)
    audit_sha = sha256_path(audit_path)
    readback_pass = bool(
        logical_sha == readback_sha
        and sha256_decompressed_csv(prediction_path)
        == prediction_artifact["decompressed_sha256"]
        and len(audit_sha) == 64
    )
    if not readback_pass:
        raise RuntimeError("exp484 Stage 1 frozen artifact SHA readback failed")
    frozen = {
        "frozen_before_truth_or_control_attachment": True,
        "rows": len(prediction),
        "wells": int(prediction["well_id"].nunique()),
        "prediction_columns": [PRIMARY_CANDIDATE],
        "logical_columns": logical_columns,
        "logical_content_sha256": logical_sha,
        "schema_sha256": dataframe_schema_sha(prediction),
        "raw_gzip_sha256": prediction_artifact["raw_sha256"],
        "decompressed_content_sha256": prediction_artifact[
            "decompressed_sha256"
        ],
        "well_audit_sha256": audit_sha,
        "sha_readback": {
            "logical_content_sha256": readback_sha,
            "well_audit_sha256": audit_sha,
            "passed": readback_pass,
        },
        "truth_access_ledger_before_freeze": ledger.report(),
    }
    ledger.mark_frozen()
    return prediction, audit, frozen, {
        "prediction": prediction_artifact,
        "well_audit": {
            "path": str(audit_path),
            "raw_sha256": audit_sha,
        },
    }


def stage1_saved_input_paths(config: Mapping[str, Any]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for key in (
        "saved_control",
        "exp209_hmm_control",
        "fold_assignment",
        "hidden_like_assignment",
    ):
        spec = dict(get_nested(config, f"data.{key}") or {})
        path = resolve_existing(
            str(spec["filename"]),
            [str(value) for value in spec.get("candidates", [])],
            [str(value) for value in spec.get("patterns", [])],
        )
        paths[key] = str(path)
    return paths


def _align_stage1_on_id(
    frame: pd.DataFrame,
    source: pd.DataFrame,
    columns: Sequence[str],
    *,
    label: str,
) -> pd.DataFrame:
    aligned_source = source.copy()
    aligned_source["id"] = aligned_source["id"].astype(str)
    if aligned_source["id"].duplicated().any():
        raise ValueError(f"{label} contains duplicate IDs")
    aligned = aligned_source.set_index("id").reindex(frame["id"].astype(str))
    if aligned[list(columns)].isna().any().any():
        raise ValueError(f"{label} has missing aligned rows")
    result = frame.copy()
    for column in columns:
        result[str(column)] = aligned[str(column)].to_numpy()
    return result


def attach_truth_late_stage1(
    prediction: pd.DataFrame,
    frozen: Mapping[str, Any],
    config: Mapping[str, Any],
    raw_dir: Path,
    ledger: TruthAccessLedger,
    saved_paths: Mapping[str, str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    require_frozen_prediction(frozen)
    ledger.require_frozen()
    logical_sha = dataframe_content_sha(
        prediction,
        list(frozen["logical_columns"]),
    )
    if logical_sha != str(frozen["logical_content_sha256"]):
        raise RuntimeError("exp484 Stage 1 candidate changed after prediction freeze")

    wells = sorted(prediction["well_id"].astype(str).unique().tolist())
    truth = load_suffix_truth_after_freeze(wells, raw_dir, ledger)
    frame = _align_stage1_on_id(
        prediction,
        truth,
        ["true_tvt"],
        label="raw suffix truth",
    )

    control_spec = dict(get_nested(config, "data.saved_control") or {})
    control_path = Path(saved_paths["saved_control"])
    if sha256_path(control_path) != str(control_spec["expected_raw_sha256"]):
        raise ValueError("exp484 saved exp404 control raw SHA mismatch")
    if sha256_decompressed_csv(control_path) != str(
        control_spec["expected_decompressed_sha256"]
    ):
        raise ValueError("exp484 saved exp404 control decompressed SHA mismatch")
    control_source = str(control_spec["prediction_column"])
    control = pd.read_csv(
        control_path,
        compression="gzip",
        usecols=["id", control_source],
        dtype={"id": str},
    )
    ledger.record_control(len(control))
    control[control_source] = restore_frozen_float32_column(
        control[control_source],
        label=f"saved exp404 {control_source}",
    )
    control = control.rename(columns={control_source: PRIMARY_CONTROL})
    frame = _align_stage1_on_id(
        frame,
        control[["id", PRIMARY_CONTROL]],
        [PRIMARY_CONTROL],
        label="saved exp404 scale-5 control",
    )

    hmm_spec = dict(get_nested(config, "data.exp209_hmm_control") or {})
    hmm_path = Path(saved_paths["exp209_hmm_control"])
    if sha256_decompressed_csv(hmm_path) != str(
        hmm_spec["expected_decompressed_sha256"]
    ):
        raise ValueError("exp484 saved exp209 HMM decompressed SHA mismatch")
    hmm_source = str(hmm_spec["prediction_column"])
    hmm = pd.read_csv(
        hmm_path,
        compression="gzip",
        usecols=["id", hmm_source],
        dtype={"id": str},
    )
    ledger.record_control(len(hmm))
    hmm[hmm_source] = pd.to_numeric(hmm[hmm_source], errors="raise")
    hmm = hmm.rename(columns={hmm_source: "saved_exp209_hmm"})
    frame = _align_stage1_on_id(
        frame,
        hmm[["id", "saved_exp209_hmm"]],
        ["saved_exp209_hmm"],
        label="saved exp209 HMM",
    )

    fold_spec = dict(get_nested(config, "data.fold_assignment") or {})
    fold_path = Path(saved_paths["fold_assignment"])
    if sha256_decompressed_csv(fold_path) != str(
        fold_spec["expected_decompressed_sha256"]
    ):
        raise ValueError("exp484 reporting-fold decompressed SHA mismatch")
    safe_columns = [str(value) for value in fold_spec["safe_columns"]]
    forbidden = {
        str(value)
        for value in fold_spec.get("forbidden_decoder_columns", [])
    }
    if set(safe_columns) != {"well_id", "row_idx", "suffix_offset", "fold"}:
        raise ValueError("exp484 fold allowlist must contain identity/fold only")
    if set(safe_columns) & forbidden:
        raise ValueError("exp484 fold allowlist contains forbidden decoder columns")
    fold = pd.read_csv(
        fold_path,
        compression="gzip",
        usecols=safe_columns,
        dtype={"well_id": str},
    )
    ledger.record_fold(len(fold))
    for column in ("row_idx", "suffix_offset", "fold"):
        fold[column] = pd.to_numeric(
            fold[column],
            errors="raise",
        ).astype(np.int64)
    if fold.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("exp484 reporting-fold identity is duplicated")
    fold = fold.rename(columns={"suffix_offset": "reporting_suffix_offset"})
    frame = frame.merge(
        fold,
        on=["well_id", "row_idx"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if frame[["fold", "reporting_suffix_offset"]].isna().any().any():
        raise ValueError("exp484 reporting-fold attachment is incomplete")
    if not np.array_equal(
        frame["suffix_offset"].to_numpy(np.int64),
        frame["reporting_suffix_offset"].to_numpy(np.int64),
    ):
        raise ValueError("exp484 reporting-fold suffix identity mismatch")
    frame = frame.drop(columns=["reporting_suffix_offset"])

    hidden_spec = dict(get_nested(config, "data.hidden_like_assignment") or {})
    hidden_path = Path(saved_paths["hidden_like_assignment"])
    if sha256_path(hidden_path) != str(hidden_spec["expected_sha256"]):
        raise ValueError("exp484 hidden-like assignment raw SHA mismatch")
    role_columns = {
        str(scope): str(column)
        for scope, column in dict(hidden_spec["role_columns"]).items()
    }
    hidden = pd.read_csv(
        hidden_path,
        usecols=["well_id", *role_columns.values()],
        dtype={"well_id": str},
    )
    ledger.record_hidden_like(len(hidden))
    if hidden["well_id"].duplicated().any():
        raise ValueError("exp484 hidden-like assignment has duplicate wells")
    for scope, column in role_columns.items():
        actual = {
            str(key): int(value)
            for key, value in hidden[column]
            .astype(str)
            .value_counts(dropna=False)
            .sort_index()
            .items()
        }
        expected = {
            str(key): int(value)
            for key, value in dict(
                hidden_spec["expected_role_counts"][scope]
            ).items()
        }
        if actual != expected:
            raise ValueError(
                f"exp484 hidden-like role counts changed for {scope}"
            )
    frame = frame.merge(
        hidden,
        on="well_id",
        how="left",
        validate="many_to_one",
    )
    if frame[list(role_columns.values())].isna().any().any():
        raise ValueError("exp484 hidden-like role attachment is incomplete")
    frame["hidden_like_spatial"] = frame[
        role_columns["hidden_like_spatial"]
    ].eq("valid")
    frame["hidden_like_typewell_purged"] = frame[
        role_columns["hidden_like_typewell_purged"]
    ].eq("valid")
    frame["candidate_hmm_50_50"] = 0.5 * (
        frame[PRIMARY_CANDIDATE].to_numpy(np.float64)
        + frame["saved_exp209_hmm"].to_numpy(np.float64)
    )
    frame["control_hmm_50_50"] = 0.5 * (
        frame[PRIMARY_CONTROL].to_numpy(np.float64)
        + frame["saved_exp209_hmm"].to_numpy(np.float64)
    )
    finite_columns = [
        "true_tvt",
        PRIMARY_CANDIDATE,
        PRIMARY_CONTROL,
        "saved_exp209_hmm",
        "candidate_hmm_50_50",
        "control_hmm_50_50",
    ]
    if not np.isfinite(frame[finite_columns].to_numpy(np.float64)).all():
        raise ValueError("exp484 Stage 1 late readout contains non-finite values")
    expected_folds = [
        int(value) for value in get_nested(config, "validation.expected_folds")
    ]
    if sorted(frame["fold"].astype(int).unique().tolist()) != expected_folds:
        raise ValueError("exp484 reporting-fold set mismatch")
    return frame, {
        "truth_attached_after_prediction_freeze": True,
        "saved_controls_attached_after_prediction_freeze": True,
        "candidate_content_sha256_reverified": logical_sha,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "folds": expected_folds,
        "saved_input_paths": dict(saved_paths),
        "truth_access_ledger": ledger.report(),
    }


def stage1_metric_record(
    frame: pd.DataFrame,
    mask: np.ndarray,
    *,
    candidate_column: str,
    control_column: str,
    comparison: str,
    scope: str,
) -> dict[str, Any]:
    selected = frame.loc[mask]
    if selected.empty:
        raise ValueError(f"exp484 Stage 1 metric scope is empty: {scope}")
    truth = selected["true_tvt"].to_numpy(np.float64)
    candidate = selected[candidate_column].to_numpy(np.float64)
    control = selected[control_column].to_numpy(np.float64)
    candidate_rmse = rmse(truth, candidate)
    control_rmse = rmse(truth, control)
    return {
        "candidate": candidate_column,
        "control": control_column,
        "comparison": comparison,
        "scope": scope,
        "rows": len(selected),
        "wells": int(selected["well_id"].nunique()),
        "candidate_rmse": candidate_rmse,
        "control_rmse": control_rmse,
        "improvement_ft": control_rmse - candidate_rmse,
        "delta_rmse_candidate_minus_control": candidate_rmse - control_rmse,
        "candidate_mae": float(np.mean(np.abs(candidate - truth))),
        "control_mae": float(np.mean(np.abs(control - truth))),
    }


def stage1_metric_scopes(
    frame: pd.DataFrame,
) -> list[tuple[str, np.ndarray]]:
    scopes: list[tuple[str, np.ndarray]] = [
        ("overall", np.ones(len(frame), dtype=bool)),
    ]
    for fold in sorted(frame["fold"].astype(int).unique().tolist()):
        scopes.append((f"fold_{fold}", frame["fold"].eq(fold).to_numpy()))
    scopes.extend(
        [
            ("raw_gr_observed", frame["raw_gr_observed"].to_numpy(bool)),
            ("raw_gr_missing", ~frame["raw_gr_observed"].to_numpy(bool)),
            (
                "missing_fraction_high",
                frame["well_missing_fraction"].ge(0.30).to_numpy(),
            ),
            ("md_since_1000_plus", frame["md_since"].ge(1000.0).to_numpy()),
            ("hidden_like_spatial", frame["hidden_like_spatial"].to_numpy(bool)),
            (
                "hidden_like_typewell_purged",
                frame["hidden_like_typewell_purged"].to_numpy(bool),
            ),
        ]
    )
    return scopes


def build_stage1_metric_outputs(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scopes = stage1_metric_scopes(frame)
    primary = pd.DataFrame(
        [
            stage1_metric_record(
                frame,
                mask,
                candidate_column=PRIMARY_CANDIDATE,
                control_column=PRIMARY_CONTROL,
                comparison="fixed_student_t_filtering_vs_saved_exp404_scale5_x1p0",
                scope=scope,
            )
            for scope, mask in scopes
        ]
    )
    blend = pd.DataFrame(
        [
            stage1_metric_record(
                frame,
                mask,
                candidate_column="candidate_hmm_50_50",
                control_column="control_hmm_50_50",
                comparison="fixed_exp209_hmm_pf_50_50",
                scope=scope,
            )
            for scope, mask in scopes
        ]
    )
    by_well_rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well_id", sort=True):
        truth = group["true_tvt"].to_numpy(np.float64)
        candidate = group[PRIMARY_CANDIDATE].to_numpy(np.float64)
        control = group[PRIMARY_CONTROL].to_numpy(np.float64)
        candidate_rmse = rmse(truth, candidate)
        control_rmse = rmse(truth, control)
        by_well_rows.append(
            {
                "well_id": str(well),
                "rows": len(group),
                "candidate_rmse": candidate_rmse,
                "control_rmse": control_rmse,
                "improvement_ft": control_rmse - candidate_rmse,
                "delta_rmse_candidate_minus_control": (
                    candidate_rmse - control_rmse
                ),
                "well_missing_fraction": float(
                    group["well_missing_fraction"].iloc[0]
                ),
            }
        )
    return primary, pd.DataFrame(by_well_rows), blend


def _stage1_scope_row(metrics: pd.DataFrame, scope: str) -> pd.Series:
    selected = metrics.loc[metrics["scope"].eq(scope)]
    if len(selected) != 1:
        raise ValueError(f"exp484 expected one Stage 1 metric row for {scope}")
    return selected.iloc[0]


def evaluate_stage1_gate(
    config: Mapping[str, Any],
    frame: pd.DataFrame,
    audit: pd.DataFrame,
    frozen: Mapping[str, Any],
    primary_metrics: pd.DataFrame,
    by_well_metrics: pd.DataFrame,
    blend_metrics: pd.DataFrame,
    ledger_at_freeze: Mapping[str, Any],
    raw_report: Mapping[str, Any],
    runtime_seconds: float,
    rss_gb: float,
) -> dict[str, Any]:
    technical_config = dict(get_nested(config, "guards.technical") or {})
    scientific_config = dict(get_nested(config, "guards.scientific") or {})
    overall = _stage1_scope_row(primary_metrics, "overall")
    blend_overall = _stage1_scope_row(blend_metrics, "overall")
    fold_rows = primary_metrics.loc[
        primary_metrics["scope"].str.startswith("fold_")
    ]
    improved_folds = int((fold_rows["improvement_ft"] > 0.0).sum())
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_folds = [
        int(value) for value in get_nested(config, "validation.expected_folds")
    ]
    before = dict(ledger_at_freeze["before_freeze"])
    control_difference = abs(
        float(overall["control_rmse"])
        - float(get_nested(config, "validation.primary_control_rmse_ft"))
    )
    blend_control_difference = abs(
        float(blend_overall["control_rmse"])
        - float(
            get_nested(
                config,
                "validation.fixed_hmm_pf_50_50_control_rmse_ft",
            )
        )
    )
    execution_counts = {
        "scientific_variants": 1,
        "candidate_pf_well_runs": int(audit["pf_well_runs"].sum()),
        "seed_well_trajectories": int(
            audit["seed_well_trajectories"].sum()
        ),
        "particle_starts": int(audit["particle_starts"].sum()),
        "control_pf_well_runs": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
    }
    expected_counts = {
        "scientific_variants": 1,
        "candidate_pf_well_runs": 773,
        "seed_well_trajectories": 98944,
        "particle_starts": 49472000,
        "control_pf_well_runs": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
    }
    fixed = dict(get_nested(config, "model.fixed_from_exp404") or {})
    seed_identity = bool(
        all(
            int(seed) == stable_seed("likpf", "train", str(well))
            for well, seed in zip(
                audit["well_id"],
                audit["seed_base"],
                strict=True,
            )
        )
    )
    technical_checks = {
        "stage0_all_technical_gates_passed": bool(
            get_nested(config, "stage_0_result.all_technical_gates_passed")
        ),
        "formula_unit_contract": bool(
            student_t_formula_contract(
                float(get_nested(config, "model.changed_factor.df"))
            )["passed"]
        ),
        "raw_input_identity": bool(
            raw_report["logical_sha256"]
            == str(get_nested(config, "data.expected_raw_well_identity_sha256"))
        ),
        "prediction_rows": len(frame) == expected_rows,
        "prediction_wells": int(frame["well_id"].nunique()) == expected_wells,
        "reporting_folds": sorted(frame["fold"].astype(int).unique().tolist())
        == expected_folds,
        "all_wells_completed": bool(
            len(audit) == expected_wells and audit["status"].eq("ok").all()
        ),
        "finite_prediction_coverage": bool(
            np.isfinite(
                frame[[PRIMARY_CANDIDATE]].to_numpy(np.float64)
            ).all()
        ),
        "stable_seed_identity": seed_identity,
        "ess_ledger_valid": bool(
            float(audit["minimum_ess_min"].min()) > 0.0
            and float(audit["minimum_ess_max"].max())
            <= float(fixed["particles"])
        ),
        "resampling_ledger_valid": bool(
            int(audit["resampling_count_total"].sum()) >= 0
            and int(audit["resampling_count_min"].min()) >= 0
        ),
        "truth_error_fold_hidden_reads_before_freeze_zero": bool(
            all(int(value) == 0 for value in before.values())
        ),
        "execution_count_match": execution_counts == expected_counts,
        "artifact_sha_readback": bool(frozen["sha_readback"]["passed"]),
        "saved_control_rmse_parity": bool(
            control_difference
            <= float(
                technical_config[
                    "require_saved_control_rmse_parity_atol_ft"
                ]
            )
        ),
        "fixed_hmm_pf_50_50_control_parity": bool(
            blend_control_difference
            <= float(
                technical_config[
                    "require_fixed_hmm_pf_50_50_parity_atol_ft"
                ]
            )
        ),
        "runtime": bool(
            runtime_seconds
            <= float(get_nested(config, "runtime.maximum_seconds"))
        ),
        "peak_rss": bool(
            rss_gb <= float(get_nested(config, "runtime.maximum_peak_rss_gb"))
        ),
    }
    technical = {
        "checks": technical_checks,
        "passed": bool(all(technical_checks.values())),
        "execution_counts": execution_counts,
        "saved_control_rmse_absolute_difference": control_difference,
        "fixed_hmm_pf_50_50_control_rmse_absolute_difference": (
            blend_control_difference
        ),
        "runtime_seconds": runtime_seconds,
        "peak_rss_gb": rss_gb,
        "truth_access_ledger_at_freeze": dict(ledger_at_freeze),
    }
    scope_rules = {
        "raw_gr_observed": (
            "minimum_gain",
            "minimum_raw_gr_observed_gain_ft",
        ),
        "raw_gr_missing": (
            "maximum_regression",
            "maximum_raw_gr_missing_regression_ft",
        ),
        "missing_fraction_high": (
            "maximum_regression",
            "maximum_high_missing_well_regression_ft",
        ),
        "md_since_1000_plus": (
            "maximum_regression",
            "maximum_long_tail_1000_plus_regression_ft",
        ),
        "hidden_like_spatial": (
            "maximum_regression",
            "maximum_hidden_like_spatial_regression_ft",
        ),
        "hidden_like_typewell_purged": (
            "maximum_regression",
            "maximum_hidden_like_typewell_purged_regression_ft",
        ),
    }
    scope_checks: dict[str, Any] = {}
    for scope, (kind, key) in scope_rules.items():
        row = _stage1_scope_row(primary_metrics, scope)
        threshold = float(scientific_config[key])
        improvement = float(row["improvement_ft"])
        delta = float(row["delta_rmse_candidate_minus_control"])
        passed = (
            improvement >= threshold
            if kind == "minimum_gain"
            else delta <= threshold
        )
        scope_checks[scope] = {
            "candidate_rmse": float(row["candidate_rmse"]),
            "control_rmse": float(row["control_rmse"]),
            "improvement_ft": improvement,
            "delta_rmse_candidate_minus_control": delta,
            "rule": kind,
            "threshold_ft": threshold,
            "passed": bool(passed),
        }
    by_well_delta = by_well_metrics["delta_rmse_candidate_minus_control"]
    by_well_p95 = float(by_well_delta.quantile(0.95))
    worst_well = float(by_well_delta.max())
    primary_gate = {
        "candidate_rmse": float(overall["candidate_rmse"]),
        "control_rmse": float(overall["control_rmse"]),
        "improvement_ft": float(overall["improvement_ft"]),
        "minimum_improvement_ft": float(
            scientific_config["minimum_pooled_rmse_gain_vs_control_ft"]
        ),
        "improved_folds": improved_folds,
        "minimum_improved_folds": int(
            scientific_config["minimum_improved_folds"]
        ),
        "scope_checks": scope_checks,
        "by_well_delta_p95_ft": by_well_p95,
        "maximum_by_well_delta_p95_ft": float(
            scientific_config["maximum_by_well_delta_p95_ft"]
        ),
        "worst_well_regression_ft": worst_well,
        "maximum_worst_well_regression_ft": float(
            scientific_config["maximum_worst_well_regression_ft"]
        ),
    }
    primary_gate["passed"] = bool(
        primary_gate["improvement_ft"]
        >= primary_gate["minimum_improvement_ft"]
        and improved_folds >= primary_gate["minimum_improved_folds"]
        and all(item["passed"] for item in scope_checks.values())
        and by_well_p95 <= primary_gate["maximum_by_well_delta_p95_ft"]
        and worst_well <= primary_gate["maximum_worst_well_regression_ft"]
    )
    blend_guard = {
        "candidate_rmse": float(blend_overall["candidate_rmse"]),
        "control_rmse": float(blend_overall["control_rmse"]),
        "delta_rmse_candidate_minus_control": float(
            blend_overall["delta_rmse_candidate_minus_control"]
        ),
        "maximum_regression_ft": float(
            scientific_config["maximum_fixed_hmm_pf_50_50_regression_ft"]
        ),
    }
    blend_guard["passed"] = bool(
        blend_guard["delta_rmse_candidate_minus_control"]
        <= blend_guard["maximum_regression_ft"]
    )
    passed = bool(
        technical["passed"]
        and primary_gate["passed"]
        and blend_guard["passed"]
    )
    return {
        "experiment": EXPERIMENT_NAME,
        "stage": "stage1_all_well_train_side_cv",
        "passed": passed,
        "decision": (
            "eligible_for_separate_raw_test_inference_design"
            if passed
            else "terminal_close_without_student_t_or_pf_rescue"
        ),
        "technical_gate": technical,
        "primary_scientific_gate": primary_gate,
        "fixed_exp209_hmm_pf_50_50_guard": blend_guard,
        "failure_action": (
            "close_without_df_scale_temperature_clip_mixture_particle_seed_"
            "transition_resampling_well_gate_blend_selector_or_same_oof_rescue"
        ),
    }


def run_stage1(config: Mapping[str, Any]) -> dict[str, Any]:
    require_kaggle_runtime()
    scientific_contract = validate_scientific_contract(
        config,
        require_run_approval=True,
    )
    if not bool(get_nested(config, "execution.run_stage_1", False)):
        raise RuntimeError("exp484 Stage 1 is not selected")
    started = time.time()
    output = artifact_dir()
    raw_dir = train_data_dir(config)
    raw_identity, raw_report = validate_raw_well_identity(config, raw_dir)
    wells = raw_identity["well_id"].astype(str).tolist()
    saved_paths = stage1_saved_input_paths(config)
    formula = student_t_formula_contract(
        float(get_nested(config, "model.changed_factor.df"))
    )
    if not bool(formula["passed"]):
        raise RuntimeError("exp484 Stage 1 formula contract failed")
    input_report = {
        "raw_train": raw_report,
        "well_ids": wells,
        "formula_contract": formula,
        "seed_namespace": ["likpf", "train", "well_id"],
        "saved_inputs": {
            key: {
                "path": value,
                "content_values_parsed_before_freeze": False,
            }
            for key, value in saved_paths.items()
        },
    }
    contract_artifact = write_json(
        output / f"{OUTPUT_PREFIX}_stage1_scientific_contract.json",
        scientific_contract,
    )
    input_artifact = write_json(
        output / f"{OUTPUT_PREFIX}_stage1_input_manifest.json",
        input_report,
    )
    ledger = TruthAccessLedger()
    warm_up_pf_kernel()
    candidate_started = time.time()
    results = Parallel(
        n_jobs=int(get_nested(config, "runtime.num_workers")),
        prefer="threads",
    )(
        delayed(decode_target_free_well)(well, raw_dir, config)
        for well in wells
    )
    prediction, audit, frozen, generated = (
        freeze_stage1_target_free_predictions(
            results,
            output,
            config,
            ledger,
        )
    )
    runtime_to_freeze = time.time() - candidate_started
    ledger_at_freeze = ledger.report()
    frame, late_report = attach_truth_late_stage1(
        prediction,
        frozen,
        config,
        raw_dir,
        ledger,
        saved_paths,
    )
    primary_metrics, by_well_metrics, blend_metrics = (
        build_stage1_metric_outputs(frame)
    )
    runtime_seconds = time.time() - started
    rss_gb = peak_rss_gb()
    gate = evaluate_stage1_gate(
        config,
        frame,
        audit,
        frozen,
        primary_metrics,
        by_well_metrics,
        blend_metrics,
        ledger_at_freeze,
        raw_report,
        runtime_seconds,
        rss_gb,
    )
    paths = {
        "truth_late_rows": (
            output / f"{OUTPUT_PREFIX}_stage1_truth_late_rows.csv.gz"
        ),
        "primary_metrics": (
            output / f"{OUTPUT_PREFIX}_stage1_primary_metrics.csv"
        ),
        "by_well_metrics": (
            output / f"{OUTPUT_PREFIX}_stage1_by_well_metrics.csv"
        ),
        "blend_metrics": (
            output
            / f"{OUTPUT_PREFIX}_stage1_fixed_hmm_pf_50_50_metrics.csv"
        ),
        "promotion_gate": (
            output / f"{OUTPUT_PREFIX}_stage1_promotion_gate.json"
        ),
        "runtime_ledger": (
            output / f"{OUTPUT_PREFIX}_stage1_runtime_ledger.json"
        ),
    }
    truth_artifact = write_deterministic_gzip_csv(
        frame,
        paths["truth_late_rows"],
    )
    primary_metrics.to_csv(paths["primary_metrics"], index=False)
    by_well_metrics.to_csv(paths["by_well_metrics"], index=False)
    blend_metrics.to_csv(paths["blend_metrics"], index=False)
    gate_artifact = write_json(paths["promotion_gate"], gate)
    runtime_artifact = write_json(
        paths["runtime_ledger"],
        {
            "runtime_seconds_to_prediction_freeze": runtime_to_freeze,
            "runtime_seconds_total": runtime_seconds,
            "peak_rss_gb": rss_gb,
            "runtime_versions": runtime_versions(),
            "kaggle_kernel_version": None,
            "kernel_version_recording": "record_from_kaggle_api_after_run",
        },
    )
    artifacts = {
        **generated,
        "scientific_contract": contract_artifact,
        "input_manifest": input_artifact,
        "truth_late_rows": truth_artifact,
        "primary_metrics": {
            "path": str(paths["primary_metrics"]),
            "raw_sha256": sha256_path(paths["primary_metrics"]),
        },
        "by_well_metrics": {
            "path": str(paths["by_well_metrics"]),
            "raw_sha256": sha256_path(paths["by_well_metrics"]),
        },
        "blend_metrics": {
            "path": str(paths["blend_metrics"]),
            "raw_sha256": sha256_path(paths["blend_metrics"]),
        },
        "promotion_gate": gate_artifact,
        "runtime_ledger": runtime_artifact,
    }
    status = (
        "stage1_all_gates_passed"
        if gate["passed"]
        else "stage1_gate_failed_terminal_close"
    )
    overall = _stage1_scope_row(primary_metrics, "overall")
    blend_overall = _stage1_scope_row(blend_metrics, "overall")
    summary = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": status,
        "stage": "stage1_all_well_train_side_cv",
        "cv": float(overall["candidate_rmse"]),
        "public_lb": None,
        "private_lb": None,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "folds": sorted(frame["fold"].astype(int).unique().tolist()),
        "counts": {
            "scientific_variants": 1,
            "candidate_pf_well_runs": int(audit["pf_well_runs"].sum()),
            "seed_well_trajectories": int(
                audit["seed_well_trajectories"].sum()
            ),
            "particle_starts": int(audit["particle_starts"].sum()),
            "control_pf_well_runs": 0,
            "hmm_well_runs": 0,
            "beam_well_runs": 0,
            "lightgbm_configs": 0,
            "trained_folds": 0,
            "boosters": 0,
            "gpu_runs": 0,
        },
        "candidate_rmse": float(overall["candidate_rmse"]),
        "saved_control_rmse": float(overall["control_rmse"]),
        "improvement_ft": float(overall["improvement_ft"]),
        "improved_folds": int(
            (
                primary_metrics.loc[
                    primary_metrics["scope"].str.startswith("fold_"),
                    "improvement_ft",
                ]
                > 0.0
            ).sum()
        ),
        "fixed_hmm_pf_50_50_candidate_rmse": float(
            blend_overall["candidate_rmse"]
        ),
        "fixed_hmm_pf_50_50_control_rmse": float(
            blend_overall["control_rmse"]
        ),
        "scientific_contract_sha256": scientific_contract[
            "scientific_contract_sha256"
        ],
        "frozen_prediction": frozen,
        "late_readout": late_report,
        "promotion_gate": gate,
        "runtime": {
            "candidate_seconds": runtime_to_freeze,
            "total_seconds": runtime_seconds,
            "peak_rss_gb": rss_gb,
            "versions": runtime_versions(),
        },
        "artifacts": artifacts,
        "deterministic_anchor": False,
        "inference": False,
        "submission": False,
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    summary_artifact = write_json(
        output / f"{OUTPUT_PREFIX}_stage1_summary.json",
        summary,
    )
    summary["artifacts"]["summary"] = summary_artifact
    write_json(metrics_output_path(), summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 11. Setup, configuration preview, and selected execution

# %%
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    SCIENTIFIC_CONTRACT = validate_scientific_contract(CONFIG)
    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "route": get_nested(CONFIG, "experiment.route"),
                "status": get_nested(CONFIG, "experiment.status"),
                "primary_control": PRIMARY_CONTROL,
                "primary_candidate": PRIMARY_CANDIDATE,
                "student_t_df": get_nested(
                    CONFIG,
                    "model.changed_factor.df",
                ),
                "active_variants": get_nested(
                    CONFIG,
                    "model.active_variants",
                ),
                "stage0_candidate_pf_well_runs": get_nested(
                    CONFIG,
                    "execution.stage_0_candidate_pf_well_runs",
                ),
                "stage0_seed_well_trajectories": get_nested(
                    CONFIG,
                    "execution.stage_0_seed_well_trajectories",
                ),
                "stage0_particle_starts": get_nested(
                    CONFIG,
                    "execution.stage_0_particle_starts",
                ),
                "stage1_candidate_pf_well_runs": get_nested(
                    CONFIG,
                    "execution.stage_1_candidate_pf_well_runs",
                ),
                "stage1_seed_well_trajectories": get_nested(
                    CONFIG,
                    "execution.stage_1_seed_well_trajectories",
                ),
                "stage1_particle_starts": get_nested(
                    CONFIG,
                    "execution.stage_1_particle_starts",
                ),
                "control_pf_well_runs": 0,
                "lightgbm_configs": 0,
                "boosters": 0,
                "gpu_runs": 0,
                "run_stage_0": get_nested(
                    CONFIG,
                    "execution.run_stage_0",
                ),
                "run_stage_1": get_nested(
                    CONFIG,
                    "execution.run_stage_1",
                ),
                "scientific_contract_sha256": SCIENTIFIC_CONTRACT[
                    "scientific_contract_sha256"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if bool(get_nested(CONFIG, "execution.run_stage_0", False)):
        STAGE0_RESULT = run_stage0(CONFIG)
    if bool(get_nested(CONFIG, "execution.run_stage_1", False)):
        STAGE1_RESULT = run_stage1(CONFIG)
    if not (
        bool(get_nested(CONFIG, "execution.run_stage_0", False))
        or bool(get_nested(CONFIG, "execution.run_stage_1", False))
    ):
        print(
            "exp484 has no selected train stage; inference and submission "
            "remain disabled pending separate approval."
        )
