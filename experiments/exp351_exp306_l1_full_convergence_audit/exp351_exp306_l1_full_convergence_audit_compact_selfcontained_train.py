# %% [markdown]
# # exp351 exp306 L1 full convergence audit train
#
# Target-free full technical audit of the only exp306 branch that passed
# Stage 0. The notebook runs the frozen L1 solver once for all 773 paired
# horizontal/typewell series and does not load truth or calculate a scientific
# score.

# %% [markdown]
# ## Contents
# 1. Imports and deterministic CPU runtime
# 2. Notebook-safe configuration, path, and SHA helpers
# 3. Frozen full-audit contract and parent-anchor guards
# 4. Target-free raw input and exp306-compatible preparation
# 5. Frozen second-order L1 solver
# 6. Full-series execution and technical status
# 7. Cross-run parity and full technical gate
# 8. Full-audit orchestration and generated artifacts
# 9. Setup and configuration preview
# 10. Run the separately approved Kaggle CPU full audit

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
import time
from collections.abc import Iterable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

EXPERIMENT_NAME = "exp351_exp306_l1_full_convergence_audit"
PARENT_EXPERIMENT = "exp306_robust_rts_l1_convergence_calibration_audit"
OUTPUT_PREFIX = EXPERIMENT_NAME
PARENT_OUTPUT_PREFIX = PARENT_EXPERIMENT
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
BRANCH_L1 = "l1_iter2000_rho1_tol1e4"
SERIES_KINDS = ("horizontal", "typewell")
PARENT_SAMPLE_SALT = "exp306-stage0-v1"


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = (
    os.environ.get("EXP351_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()
)


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


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def get_nested(config: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
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
    raise FileNotFoundError(f"exp351 config not found in {[str(path) for path in candidates]}")


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
    return project_root() / str(get_nested(config, "data.train_dir") or "data/raw/train")


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


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decompressed_sha256(path: Path) -> str:
    with gzip.open(path, "rb") as file_pointer:
        payload = file_pointer.read()
    return hashlib.sha256(payload).hexdigest()


def mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def dataframe_content_sha(
    frame: pd.DataFrame,
    columns: Iterable[str] | None = None,
) -> str:
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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def write_csv_plain(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return {
        "path": str(path),
        "rows": len(frame),
        "raw_sha256": sha256_path(path),
        "content_sha256": dataframe_content_sha(frame),
    }


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


def runtime_manifest() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


# %% [markdown]
# ## 3. Frozen full-audit contract and parent-anchor guards


# %%
def _require_equal(config: Mapping[str, Any], dotted_key: str, expected: Any) -> None:
    actual = get_nested(config, dotted_key)
    if actual != expected:
        raise ValueError(
            f"exp351 contract mismatch: {dotted_key}={actual!r}, expected {expected!r}"
        )


def validate_technical_contract(
    config: Mapping[str, Any],
    *,
    require_run_approval: bool = False,
) -> None:
    fixed_values = {
        "experiment.name": EXPERIMENT_NAME,
        "experiment.route": "pf_beam",
        "lineage.parent": PARENT_EXPERIMENT,
        "implementation.enabled": True,
        "implementation.scope": "l1_full_technical_audit_only",
        "implementation.canonical_notebook_adopted": True,
        "implementation.full_audit_implemented": True,
        "implementation.scientific_score_implemented": False,
        "validation.strategy": "target_free_l1_full_convergence_technical_audit",
        "validation.metric": "solver_technical_coverage",
        "validation.n_folds": 0,
        "validation.score_rows": "no_truth_no_scientific_score",
        "validation.expected_wells": 773,
        "validation.expected_series": 1546,
        "data.horizontal_allowed_columns": ["MD", "GR", "TVT_input"],
        "data.typewell_required_columns": ["TVT", "GR"],
        "parent_anchor.experiment": PARENT_EXPERIMENT,
        "parent_anchor.kaggle_kernel_version": 1,
        "parent_anchor.kaggle_kernel_id_no": 128231380,
        "model.lightgbm_config_count": 0,
        "model.fold_training_count": 0,
        "model.booster_count": 0,
        "model.parent_control_retraining": False,
        "model.l1_trend.branch_name": BRANCH_L1,
        "model.l1_trend.rho": 1.0,
        "model.l1_trend.maximum_iterations": 2000,
        "model.l1_trend.absolute_tolerance": 1.0e-4,
        "model.l1_trend.relative_tolerance": 1.0e-4,
        "model.l1_trend.adaptive_rho": False,
        "model.l1_trend.warm_start_alternative": False,
        "model.l1_trend.additional_grid": False,
        "model.robust_rts.enabled": False,
        "audit.full.active_branches": [BRANCH_L1],
        "audit.full.expected_wells": 773,
        "audit.full.series_kinds": ["horizontal", "typewell"],
        "audit.full.expected_series": 1546,
        "audit.full.runtime_limit_seconds": 30600,
        "audit.cross_run_parity.parent_sample_wells": 64,
        "audit.cross_run_parity.parent_parity_wells": 8,
        "audit.cross_run_parity.full_solver_rerun": False,
        "runtime.num_workers": 1,
        "runtime.blas_threads": 1,
        "runtime.kaggle.enable_gpu": False,
        "runtime.kaggle.enable_internet": False,
        "execution.design_approved": True,
        "execution.implementation_approved": True,
        "inference.enabled": False,
        "inference.create_submission": False,
        "inference.mode": "disabled_train_side_solver_audit_only",
    }
    for dotted_key, expected in fixed_values.items():
        _require_equal(config, dotted_key, expected)

    forbidden_flags = (
        "execution.run_scientific_score",
        "execution.run_inference",
        "execution.create_submission",
    )
    enabled_forbidden = [key for key in forbidden_flags if bool(get_nested(config, key))]
    if enabled_forbidden:
        raise RuntimeError(
            "exp351 full audit keeps scientific/inference/submission paths fail-closed: "
            f"{enabled_forbidden}"
        )
    expected_counts = {
        "audit.execution_counts.active_branches": 1,
        "audit.execution_counts.l1_solver_series_runs": 1546,
        "audit.execution_counts.stage0_control_series_runs": 0,
        "audit.execution_counts.full_rerun_series_runs": 0,
        "audit.execution_counts.parity_rerun_series_runs": 0,
        "audit.execution_counts.model_count": 0,
        "audit.execution_counts.lightgbm_config_count": 0,
        "audit.execution_counts.trained_fold_count": 0,
        "audit.execution_counts.hmm_well_runs": 0,
        "audit.execution_counts.pf_well_runs": 0,
        "audit.execution_counts.beam_well_runs": 0,
        "audit.execution_counts.booster_count": 0,
        "audit.execution_counts.parent_control_retraining": 0,
        "audit.execution_counts.gpu_runs": 0,
    }
    for dotted_key, expected in expected_counts.items():
        _require_equal(config, dotted_key, expected)
    if require_run_approval:
        if not bool(get_nested(config, "execution.kaggle_push_approved")):
            raise RuntimeError("exp351 Kaggle package/push/run is not approved")
        if not bool(get_nested(config, "execution.run_full_l1")):
            raise RuntimeError("exp351 full L1 run flag is not enabled")


def build_technical_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_technical_contract(config)
    contract: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "parent": PARENT_EXPERIMENT,
        "parent_kernel": {
            "id": get_nested(config, "parent_anchor.kaggle_kernel_id"),
            "version": get_nested(config, "parent_anchor.kaggle_kernel_version"),
            "id_no": get_nested(config, "parent_anchor.kaggle_kernel_id_no"),
        },
        "truth_or_scientific_score_loaded": False,
        "prediction_created": False,
        "submission_created": False,
        "raw_well_identity_expected_sha256": get_nested(
            config, "data.expected_raw_well_identity_sha256"
        ),
        "full_audit": {
            "branch": BRANCH_L1,
            "wells": 773,
            "series_kinds": list(SERIES_KINDS),
            "series_runs": 1546,
            "full_solver_rerun": False,
            "parity_rerun": False,
            "runtime_limit_seconds": 30600,
        },
        "l1_solver": {
            "objective": "0.5_l2_data_plus_lambda_l1_second_difference",
            "lambda": (
                "mad_first_difference_div_0.67448975_div_sqrt2"
                "_times_sqrt_2_log_n"
            ),
            "rho": 1.0,
            "maximum_iterations": 2000,
            "absolute_tolerance": 1.0e-4,
            "relative_tolerance": 1.0e-4,
            "adaptive_rho": False,
        },
        "forbidden": [
            "horizontal_TVT",
            "truth",
            "error_target",
            "formation",
            "MRR",
            "top3",
            "RMSE",
            "scientific_score",
            "RTS",
            "HMM",
            "PF",
            "Beam",
            "prediction",
            "submission",
            "solver_grid",
        ],
        "runtime_contract": {
            "num_workers": 1,
            "blas_threads": 1,
            "gpu": False,
            "internet": False,
        },
    }
    contract["scientific_contract_sha256"] = mapping_sha256(contract)
    return contract


def parent_artifact_filenames() -> dict[str, str]:
    return {
        "scientific_contract": f"{PARENT_OUTPUT_PREFIX}_scientific_contract.json",
        "stage0_gate": f"{PARENT_OUTPUT_PREFIX}_stage0_gate.json",
        "stage0_summary": f"{PARENT_OUTPUT_PREFIX}_summary.json",
        "sample_manifest": f"{PARENT_OUTPUT_PREFIX}_stage0_sample_manifest.csv",
        "stage0_input": f"{PARENT_OUTPUT_PREFIX}_stage0_input.csv.gz",
        "stage0_output": f"{PARENT_OUTPUT_PREFIX}_stage0_output.csv.gz",
        "stage0_status": f"{PARENT_OUTPUT_PREFIX}_stage0_solver_status.csv.gz",
        "parity_manifest": f"{PARENT_OUTPUT_PREFIX}_parity_manifest.json",
    }


def resolve_parent_artifact_dir(
    config: Mapping[str, Any],
    search_root: Path | None = None,
) -> Path:
    root = search_root or KAGGLE_INPUT_ROOT
    filename = parent_artifact_filenames()["scientific_contract"]
    expected_sha = str(
        get_nested(config, "parent_anchor.scientific_contract_file_sha256")
    )
    matches = [
        path
        for path in sorted(root.rglob(filename))
        if path.is_file() and sha256_path(path) == expected_sha
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            "expected exactly one SHA-matching exp306 version-1 artifact directory, "
            f"found {len(matches)} under {root}"
        )
    return matches[0].parent


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def validate_parent_anchors(
    parent_dir: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    filenames = parent_artifact_filenames()
    paths = {key: parent_dir / name for key, name in filenames.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"exp306 parent artifacts are incomplete: {missing}")

    expected_file_shas = {
        "scientific_contract": str(
            get_nested(config, "parent_anchor.scientific_contract_file_sha256")
        ),
        "stage0_gate": str(get_nested(config, "parent_anchor.stage0_gate_file_sha256")),
        "stage0_summary": str(
            get_nested(config, "parent_anchor.stage0_summary_file_sha256")
        ),
        "sample_manifest": str(
            get_nested(config, "parent_anchor.sample_manifest_raw_sha256")
        ),
    }
    actual_file_shas = {
        key: sha256_path(paths[key]) for key in expected_file_shas
    }
    mismatched_files = {
        key: {"actual": actual_file_shas[key], "expected": expected}
        for key, expected in expected_file_shas.items()
        if actual_file_shas[key] != expected
    }
    if mismatched_files:
        raise ValueError(f"exp306 parent file SHA mismatch: {mismatched_files}")

    contract = _load_json_object(paths["scientific_contract"])
    contract_without_sha = deepcopy(contract)
    stored_contract_sha = contract_without_sha.pop("scientific_contract_sha256", None)
    expected_contract_sha = str(
        get_nested(config, "parent_anchor.scientific_contract_content_sha256")
    )
    actual_contract_sha = mapping_sha256(contract_without_sha)
    if stored_contract_sha != expected_contract_sha or actual_contract_sha != expected_contract_sha:
        raise ValueError(
            "exp306 scientific contract content SHA mismatch: "
            f"stored={stored_contract_sha}, actual={actual_contract_sha}, "
            f"expected={expected_contract_sha}"
        )

    sample = pd.read_csv(paths["sample_manifest"])
    expected_sample_columns = [
        "sample_rank",
        "well_id",
        "sample_sha256",
        "horizontal_raw_sha256",
        "typewell_raw_sha256",
    ]
    if list(sample.columns) != expected_sample_columns:
        raise ValueError(
            f"exp306 sample schema mismatch: {list(sample.columns)}"
        )
    sample_content_sha = dataframe_content_sha(sample)
    expected_sample_content_sha = str(
        get_nested(config, "parent_anchor.sample_manifest_content_sha256")
    )
    if sample_content_sha != expected_sample_content_sha:
        raise ValueError(
            "exp306 sample content SHA mismatch: "
            f"{sample_content_sha} != {expected_sample_content_sha}"
        )

    gate = _load_json_object(paths["stage0_gate"])
    summary = _load_json_object(paths["stage0_summary"])
    parity = _load_json_object(paths["parity_manifest"])
    l1_gate = get_nested(gate, f"branches.{BRANCH_L1}")
    if not isinstance(l1_gate, Mapping):
        raise ValueError("exp306 Stage 0 gate has no frozen L1 branch")
    expected_logical = {
        "raw_well_identity": str(
            get_nested(config, "data.expected_raw_well_identity_sha256")
        ),
        "stage0_input": str(
            get_nested(config, "parent_anchor.stage0_input_content_sha256")
        ),
        "stage0_output": str(
            get_nested(config, "parent_anchor.stage0_l1_output_content_sha256")
        ),
        "stage0_status": str(
            get_nested(config, "parent_anchor.stage0_l1_status_content_sha256")
        ),
    }
    actual_logical = {
        "raw_well_identity": gate.get("raw_well_identity_content_sha256"),
        "stage0_input": get_nested(gate, "input_artifact.content_sha256"),
        "stage0_output": l1_gate.get("output_content_sha256"),
        "stage0_status": l1_gate.get("status_content_sha256"),
    }
    if actual_logical != expected_logical:
        raise ValueError(
            f"exp306 Stage 0 logical anchor mismatch: {actual_logical} != {expected_logical}"
        )
    if not bool(l1_gate.get("full_eligible")):
        raise ValueError("exp306 L1 branch is not marked full-eligible")
    if gate.get("truth_or_scientific_score_loaded") is not False:
        raise ValueError("exp306 parent gate violates the target-free contract")
    if gate.get("full_audit_executed") is not False:
        raise ValueError("exp306 parent unexpectedly contains a full audit")
    if summary.get("scientific_score") is not None or summary.get("submission") is not None:
        raise ValueError("exp306 parent summary contains forbidden scientific output")
    if summary.get("full_eligible_branches") != [BRANCH_L1]:
        raise ValueError("exp306 parent full-eligible branch set changed")

    parity_rows = parity.get("branches")
    if not isinstance(parity_rows, list):
        raise ValueError("exp306 parity manifest has no branch list")
    l1_parity = next(
        (row for row in parity_rows if row.get("branch") == BRANCH_L1),
        None,
    )
    if not isinstance(l1_parity, Mapping) or not bool(l1_parity.get("exact_identity")):
        raise ValueError("exp306 L1 parity is absent or not exact")
    expected_parity = {
        "output_content_sha256": str(
            get_nested(config, "parent_anchor.parity_output_content_sha256")
        ),
        "status_content_sha256": str(
            get_nested(config, "parent_anchor.parity_status_content_sha256")
        ),
        "iteration_content_sha256": str(
            get_nested(config, "parent_anchor.parity_iteration_content_sha256")
        ),
    }
    if l1_parity.get("main") != expected_parity or l1_parity.get("rerun") != expected_parity:
        raise ValueError("exp306 L1 parity hashes differ from the frozen version-1 anchor")

    for key, gate_key in (
        ("stage0_input", "input_artifact"),
        ("stage0_output", "output_artifact"),
        ("stage0_status", "solver_status_artifact"),
    ):
        artifact = get_nested(gate, gate_key)
        if not isinstance(artifact, Mapping):
            raise ValueError(f"exp306 gate lacks {gate_key}")
        if sha256_path(paths[key]) != artifact.get("raw_sha256"):
            raise ValueError(f"exp306 {key} raw gzip SHA differs from Stage 0 gate")
        if decompressed_sha256(paths[key]) != artifact.get("decompressed_sha256"):
            raise ValueError(f"exp306 {key} decompressed SHA differs from Stage 0 gate")

    manifest = {
        "parent_experiment": PARENT_EXPERIMENT,
        "kaggle_kernel_id": get_nested(config, "parent_anchor.kaggle_kernel_id"),
        "kaggle_kernel_version": get_nested(
            config, "parent_anchor.kaggle_kernel_version"
        ),
        "kaggle_kernel_id_no": get_nested(
            config, "parent_anchor.kaggle_kernel_id_no"
        ),
        "parent_artifact_dir": str(parent_dir),
        "file_sha256": actual_file_shas,
        "logical_content_sha256": actual_logical,
        "parity_content_sha256": expected_parity,
        "all_parent_anchors_match": True,
    }
    return manifest, sample


# %% [markdown]
# ## 4. Target-free raw input and exp306-compatible preparation


# %%
def validate_horizontal_target_free_frame(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> None:
    allowed = [str(value) for value in get_nested(config, "data.horizontal_allowed_columns")]
    forbidden = {
        str(value).casefold()
        for value in get_nested(config, "data.horizontal_forbidden_columns")
    }
    forbidden.update(
        {"truth", "tvt_true", "target", "mrr", "top3", "rmse", "score", "prediction"}
    )
    actual_casefold = {str(column).casefold() for column in frame.columns}
    leaked = sorted(actual_casefold & forbidden)
    if leaked:
        raise ValueError(f"target-free horizontal frame exposes forbidden columns: {leaked}")
    if list(frame.columns) != allowed:
        raise ValueError(
            f"target-free horizontal frame must expose exactly {allowed}, got {list(frame.columns)}"
        )


def validate_typewell_frame(frame: pd.DataFrame, config: Mapping[str, Any]) -> None:
    required = [str(value) for value in get_nested(config, "data.typewell_required_columns")]
    if list(frame.columns) != required:
        raise ValueError(
            f"typewell frame must expose exactly {required}, got {list(frame.columns)}"
        )


def load_horizontal_target_free(path: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    allowed = [str(value) for value in get_nested(config, "data.horizontal_allowed_columns")]
    frame = pd.read_csv(path, usecols=allowed)
    frame = frame[allowed]
    validate_horizontal_target_free_frame(frame, config)
    return frame


def load_typewell_target_free(path: Path, config: Mapping[str, Any]) -> pd.DataFrame:
    required = [str(value) for value in get_nested(config, "data.typewell_required_columns")]
    frame = pd.read_csv(path, usecols=required)
    frame = frame[required]
    validate_typewell_frame(frame, config)
    return frame


def enumerate_paired_wells(raw_dir: Path) -> list[str]:
    horizontal = {
        path.name.removesuffix("__horizontal_well.csv")
        for path in raw_dir.glob("*__horizontal_well.csv")
    }
    typewell = {
        path.name.removesuffix("__typewell.csv")
        for path in raw_dir.glob("*__typewell.csv")
    }
    if horizontal != typewell:
        raise ValueError(
            "horizontal/typewell identities differ: "
            f"missing_typewell={sorted(horizontal - typewell)[:5]}, "
            f"missing_horizontal={sorted(typewell - horizontal)[:5]}"
        )
    return sorted(horizontal)


def stable_parent_sample(
    well_ids: Iterable[str],
    *,
    sample_wells: int,
) -> pd.DataFrame:
    unique = sorted({str(well_id) for well_id in well_ids})
    if len(unique) < sample_wells:
        raise ValueError(f"cannot sample {sample_wells} wells from {len(unique)}")
    rows = []
    for well_id in unique:
        sample_sha = hashlib.sha256(
            f"{PARENT_SAMPLE_SALT}|{well_id}".encode()
        ).hexdigest()
        rows.append({"well_id": well_id, "sample_sha256": sample_sha})
    sample = (
        pd.DataFrame(rows)
        .sort_values(["sample_sha256", "well_id"], kind="mergesort")
        .head(sample_wells)
        .reset_index(drop=True)
    )
    sample.insert(0, "sample_rank", np.arange(1, len(sample) + 1, dtype=np.int64))
    return sample


def raw_well_identity_manifest(raw_dir: Path, well_ids: Iterable[str]) -> pd.DataFrame:
    rows = []
    for well_id in sorted(str(value) for value in well_ids):
        horizontal_path = raw_dir / f"{well_id}__horizontal_well.csv"
        typewell_path = raw_dir / f"{well_id}__typewell.csv"
        if not horizontal_path.exists() or not typewell_path.exists():
            raise FileNotFoundError(f"missing paired files for {well_id}")
        rows.append(
            {
                "well_id": well_id,
                "horizontal_raw_sha256": sha256_path(horizontal_path),
                "typewell_raw_sha256": sha256_path(typewell_path),
            }
        )
    return pd.DataFrame(rows).sort_values("well_id", kind="mergesort").reset_index(drop=True)


def validate_parent_sample_against_raw(
    parent_sample: pd.DataFrame,
    raw_identity: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    expected_count = int(
        get_nested(config, "audit.cross_run_parity.parent_sample_wells")
    )
    reconstructed = stable_parent_sample(
        raw_identity["well_id"].astype(str),
        sample_wells=expected_count,
    ).merge(raw_identity, on="well_id", how="left", validate="one_to_one")
    expected_content_sha = str(
        get_nested(config, "parent_anchor.sample_manifest_content_sha256")
    )
    reconstructed_sha = dataframe_content_sha(reconstructed)
    exact_frame = reconstructed.equals(parent_sample)
    if not exact_frame or reconstructed_sha != expected_content_sha:
        raise ValueError(
            "raw train does not reconstruct the frozen exp306 Stage 0 sample exactly"
        )
    return {
        "sample_wells": expected_count,
        "sample_manifest_content_sha256": reconstructed_sha,
        "sample_manifest_exact_frame": True,
    }


def prepare_gr_inputs(
    horizontal_without_truth: pd.DataFrame,
    typewell: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    validate_horizontal_target_free_frame(horizontal_without_truth, config)
    validate_typewell_frame(typewell, config)

    horizontal = horizontal_without_truth.copy()
    horizontal["MD"] = pd.to_numeric(horizontal["MD"], errors="raise")
    horizontal["TVT_input"] = pd.to_numeric(horizontal["TVT_input"], errors="coerce")
    horizontal_gr_observed = pd.to_numeric(horizontal["GR"], errors="coerce")
    horizontal_original_missing = horizontal_gr_observed.isna().to_numpy(bool)
    horizontal_md = horizontal["MD"].to_numpy(np.float64)
    if len(horizontal) < 2 or not np.isfinite(horizontal_md).all():
        raise ValueError("horizontal MD must contain at least two finite rows")
    if bool((np.diff(horizontal_md) < 0.0).any()):
        raise ValueError("horizontal rows must already be in non-decreasing MD order")
    visible_tvt_input = horizontal["TVT_input"].dropna().to_numpy(np.float64)
    if not np.isfinite(visible_tvt_input).all():
        raise ValueError("visible TVT_input rows must be finite")

    tw = typewell.copy()
    tw["TVT"] = pd.to_numeric(tw["TVT"], errors="coerce")
    tw["GR"] = pd.to_numeric(tw["GR"], errors="coerce")
    tw = tw.dropna(subset=["TVT"]).sort_values("TVT", kind="mergesort").reset_index(drop=True)
    typewell_original_missing = tw["GR"].isna().to_numpy(bool)
    tw["GR"] = tw["GR"].ffill().bfill()
    if len(tw) < 2 or not np.isfinite(tw[["TVT", "GR"]].to_numpy(np.float64)).all():
        raise ValueError("typewell requires at least two finite TVT/GR rows")
    typewell_tvt = tw["TVT"].to_numpy(np.float64)
    typewell_gr = tw["GR"].to_numpy(np.float64)
    if bool((np.diff(typewell_tvt) < 0.0).any()):
        raise ValueError("typewell TVT must be non-decreasing after stable sort")

    gr_fill = float(np.nanmean(typewell_gr))
    horizontal_gr = (
        horizontal_gr_observed.interpolate(limit_direction="both")
        .fillna(gr_fill)
        .to_numpy(np.float64)
    )
    if not np.isfinite(horizontal_gr).all():
        raise ValueError("common horizontal GR interpolation must be finite")
    return {
        "horizontal_coordinate": horizontal_md,
        "horizontal_gr": horizontal_gr,
        "horizontal_original_missing": horizontal_original_missing,
        "typewell_coordinate": typewell_tvt,
        "typewell_gr": typewell_gr,
        "typewell_original_missing": typewell_original_missing,
    }


def load_prepared_full(
    raw_dir: Path,
    well_ids: Iterable[str],
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    prepared_by_well: dict[str, dict[str, Any]] = {}
    for well_id in sorted(str(value) for value in well_ids):
        horizontal = load_horizontal_target_free(
            raw_dir / f"{well_id}__horizontal_well.csv", config
        )
        typewell = load_typewell_target_free(
            raw_dir / f"{well_id}__typewell.csv", config
        )
        prepared_by_well[well_id] = prepare_gr_inputs(horizontal, typewell, config)
    return prepared_by_well


def build_input_frame(prepared_by_well: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for well_id in sorted(prepared_by_well):
        prepared = prepared_by_well[well_id]
        for series_kind in SERIES_KINDS:
            coordinate = np.asarray(prepared[f"{series_kind}_coordinate"], dtype=np.float64)
            values = np.asarray(prepared[f"{series_kind}_gr"], dtype=np.float64)
            parts.append(
                pd.DataFrame(
                    {
                        "well_id": well_id,
                        "series_kind": series_kind,
                        "position": np.arange(len(values), dtype=np.int64),
                        "coordinate": coordinate,
                        "input_gr": values,
                        "original_missing": np.asarray(
                            prepared[f"{series_kind}_original_missing"], dtype=bool
                        ),
                    }
                )
            )
    if not parts:
        raise ValueError("full audit has no prepared input series")
    return pd.concat(parts, ignore_index=True).sort_values(
        ["well_id", "series_kind", "position"], kind="mergesort"
    ).reset_index(drop=True)


# %% [markdown]
# ## 5. Frozen second-order L1 solver


# %%
def robust_scale(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return 0.0
    center = float(np.median(finite))
    return float(np.median(np.abs(finite - center)) / 0.67448975)


def second_difference(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    return x[:-2] - 2.0 * x[1:-1] + x[2:]


def second_difference_transpose(values: np.ndarray, size: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    output = np.zeros(size, dtype=np.float64)
    output[:-2] += source
    output[1:-1] -= 2.0 * source
    output[2:] += source
    return output


def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.sign(array) * np.maximum(np.abs(array) - float(threshold), 0.0)


def l1_trend_smooth(
    values: np.ndarray,
    spec: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1 or len(y) < 2 or not np.isfinite(y).all():
        raise ValueError("L1 trend filtering requires a finite series with at least two rows")
    noise_std = robust_scale(np.diff(y)) / math.sqrt(2.0)
    regularization = float(noise_std * math.sqrt(2.0 * math.log(len(y))))
    if len(y) < 3 or regularization == 0.0:
        return y.copy(), {
            "converged": True,
            "iterations": 0,
            "lambda": regularization,
            "rho": float(spec["rho"]),
            "primal_residual": 0.0,
            "dual_residual": 0.0,
            "finite_output": True,
        }
    try:
        from scipy.linalg import cho_solve_banded, cholesky_banded
    except ImportError as error:
        raise RuntimeError(
            "SciPy is unavailable; l1_trend must technical-fail without fallback"
        ) from error

    rho = float(spec["rho"])
    maximum_iterations = int(spec["maximum_iterations"])
    absolute_tolerance = float(spec["absolute_tolerance"])
    relative_tolerance = float(spec["relative_tolerance"])
    n = len(y)
    m = n - 2
    diagonal = np.ones(n, dtype=np.float64)
    diagonal[:-2] += rho
    diagonal[1:-1] += 4.0 * rho
    diagonal[2:] += rho
    first_upper = np.zeros(n - 1, dtype=np.float64)
    first_upper[:-1] -= 2.0 * rho
    first_upper[1:] -= 2.0 * rho
    second_upper = np.full(n - 2, rho, dtype=np.float64)
    banded = np.zeros((3, n), dtype=np.float64)
    banded[2] = diagonal
    banded[1, 1:] = first_upper
    banded[0, 2:] = second_upper
    factor = cholesky_banded(banded, lower=False, check_finite=False)

    x = y.copy()
    z = second_difference(x)
    dual = np.zeros(m, dtype=np.float64)
    converged = False
    primal_norm = math.inf
    dual_norm = math.inf
    iteration = 0
    for _iteration in range(1, maximum_iterations + 1):
        iteration = _iteration
        right_hand_side = y + rho * second_difference_transpose(z - dual, n)
        x = cho_solve_banded((factor, False), right_hand_side, check_finite=False)
        d2x = second_difference(x)
        previous_z = z.copy()
        z = soft_threshold(d2x + dual, regularization / rho)
        dual = dual + d2x - z
        primal_norm = float(np.linalg.norm(d2x - z))
        dual_norm = float(
            np.linalg.norm(rho * second_difference_transpose(z - previous_z, n))
        )
        primal_tolerance = math.sqrt(m) * absolute_tolerance + relative_tolerance * max(
            np.linalg.norm(d2x), np.linalg.norm(z)
        )
        dual_tolerance = math.sqrt(n) * absolute_tolerance + relative_tolerance * float(
            np.linalg.norm(rho * second_difference_transpose(dual, n))
        )
        if primal_norm <= primal_tolerance and dual_norm <= dual_tolerance:
            converged = True
            break
    return np.asarray(x, dtype=np.float64), {
        "converged": converged,
        "iterations": iteration,
        "lambda": regularization,
        "rho": rho,
        "primal_residual": primal_norm,
        "dual_residual": dual_norm,
        "finite_output": bool(np.isfinite(x).all()),
    }


def l1_spec(config: Mapping[str, Any]) -> dict[str, Any]:
    return dict(get_nested(config, "model.l1_trend") or {})


# %% [markdown]
# ## 6. Full-series execution and technical status


# %%
def run_l1_full(
    prepared_by_well: Mapping[str, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    output_parts: list[pd.DataFrame] = []
    status_rows: list[dict[str, Any]] = []
    spec = l1_spec(config)
    started = time.perf_counter()
    for well_id in sorted(prepared_by_well):
        prepared = prepared_by_well[well_id]
        for series_kind in SERIES_KINDS:
            values = np.asarray(prepared[f"{series_kind}_gr"], dtype=np.float64)
            coordinate = np.asarray(
                prepared[f"{series_kind}_coordinate"], dtype=np.float64
            )
            try:
                output, details = l1_trend_smooth(values, spec)
                variance = np.full(len(output), np.nan, dtype=np.float64)
                length_match = len(output) == len(values)
                finite_output = bool(length_match and np.isfinite(output).all())
                converged = bool(details.get("converged", False))
                technical_pass = bool(
                    np.isfinite(values).all()
                    and length_match
                    and finite_output
                    and converged
                )
                error = ""
            except Exception as exception:
                output = np.full(len(values), np.nan, dtype=np.float64)
                variance = np.full(len(values), np.nan, dtype=np.float64)
                details = {"converged": False, "iterations": 0, "finite_output": False}
                length_match = True
                finite_output = False
                converged = False
                technical_pass = False
                error = f"{type(exception).__name__}: {exception}"[:500]

            output_parts.append(
                pd.DataFrame(
                    {
                        "branch": BRANCH_L1,
                        "well_id": well_id,
                        "series_kind": series_kind,
                        "position": np.arange(len(values), dtype=np.int64),
                        "coordinate": coordinate,
                        "input_gr": values,
                        "output_gr": np.asarray(output, dtype=np.float64),
                        "posterior_variance": variance,
                    }
                )
            )
            status_rows.append(
                {
                    "branch": BRANCH_L1,
                    "well_id": well_id,
                    "series_kind": series_kind,
                    "rows": len(values),
                    "finite_input": bool(np.isfinite(values).all()),
                    "length_match": bool(length_match),
                    "order_match": True,
                    "finite_output": bool(finite_output),
                    "converged": bool(converged),
                    "silent_fallback": False,
                    "technical_pass": bool(technical_pass),
                    "iterations": int(details.get("iterations", 0)),
                    "relative_mean_change": details.get("relative_mean_change", np.nan),
                    "measurement_std": details.get("measurement_std", np.nan),
                    "acceleration_std": details.get("acceleration_std", np.nan),
                    "minimum_weight": details.get("minimum_weight", np.nan),
                    "lambda": details.get("lambda", np.nan),
                    "rho": details.get("rho", np.nan),
                    "primal_residual": details.get("primal_residual", np.nan),
                    "dual_residual": details.get("dual_residual", np.nan),
                    "error": error,
                }
            )
    elapsed = float(time.perf_counter() - started)
    output_frame = pd.concat(output_parts, ignore_index=True).sort_values(
        ["branch", "well_id", "series_kind", "position"], kind="mergesort"
    ).reset_index(drop=True)
    status_frame = pd.DataFrame(status_rows).sort_values(
        ["branch", "well_id", "series_kind"], kind="mergesort"
    ).reset_index(drop=True)
    return output_frame, status_frame, elapsed


def output_matches_input(input_frame: pd.DataFrame, output_frame: pd.DataFrame) -> bool:
    keys = ["well_id", "series_kind", "position"]
    expected = input_frame.reset_index(drop=True)
    actual = output_frame.reset_index(drop=True)
    if len(expected) != len(actual) or actual.duplicated(keys).any():
        return False
    if not expected[keys].equals(actual[keys]):
        return False
    return bool(
        np.array_equal(
            expected["coordinate"].to_numpy(np.float64),
            actual["coordinate"].to_numpy(np.float64),
        )
        and np.array_equal(
            expected["input_gr"].to_numpy(np.float64),
            actual["input_gr"].to_numpy(np.float64),
        )
    )


# %% [markdown]
# ## 7. Cross-run parity and full technical gate


# %%
def parity_content_hashes(
    output_frame: pd.DataFrame,
    status_frame: pd.DataFrame,
) -> dict[str, str]:
    output_sorted = output_frame.sort_values(
        ["branch", "well_id", "series_kind", "position"], kind="mergesort"
    ).reset_index(drop=True)
    status_sorted = status_frame.sort_values(
        ["branch", "well_id", "series_kind"], kind="mergesort"
    ).reset_index(drop=True)
    iteration_columns = ["branch", "well_id", "series_kind", "iterations"]
    return {
        "output_content_sha256": dataframe_content_sha(output_sorted),
        "status_content_sha256": dataframe_content_sha(status_sorted),
        "iteration_content_sha256": dataframe_content_sha(
            status_sorted[iteration_columns]
        ),
    }


def evaluate_cross_run_parity(
    input_frame: pd.DataFrame,
    output_frame: pd.DataFrame,
    status_frame: pd.DataFrame,
    parent_sample: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    sample_count = int(
        get_nested(config, "audit.cross_run_parity.parent_sample_wells")
    )
    parity_count = int(
        get_nested(config, "audit.cross_run_parity.parent_parity_wells")
    )
    ranked_sample = parent_sample.sort_values("sample_rank", kind="mergesort")
    sample_wells = ranked_sample["well_id"].astype(str).head(sample_count).tolist()
    parity_wells = sample_wells[:parity_count]
    input_subset = input_frame.loc[input_frame["well_id"].isin(sample_wells)].copy()
    input_subset = input_subset.sort_values(
        ["well_id", "series_kind", "position"], kind="mergesort"
    ).reset_index(drop=True)
    output_subset = output_frame.loc[output_frame["well_id"].isin(sample_wells)].copy()
    status_subset = status_frame.loc[status_frame["well_id"].isin(sample_wells)].copy()
    sample_hashes = {
        "input_content_sha256": dataframe_content_sha(input_subset),
        "output_content_sha256": dataframe_content_sha(
            output_subset.sort_values(
                ["branch", "well_id", "series_kind", "position"], kind="mergesort"
            ).reset_index(drop=True)
        ),
        "status_content_sha256": dataframe_content_sha(
            status_subset.sort_values(
                ["branch", "well_id", "series_kind"], kind="mergesort"
            ).reset_index(drop=True)
        ),
    }
    expected_sample_hashes = {
        "input_content_sha256": str(
            get_nested(config, "parent_anchor.stage0_input_content_sha256")
        ),
        "output_content_sha256": str(
            get_nested(config, "parent_anchor.stage0_l1_output_content_sha256")
        ),
        "status_content_sha256": str(
            get_nested(config, "parent_anchor.stage0_l1_status_content_sha256")
        ),
    }
    parity_output = output_frame.loc[
        output_frame["well_id"].isin(parity_wells)
    ].copy()
    parity_status = status_frame.loc[
        status_frame["well_id"].isin(parity_wells)
    ].copy()
    parity_hashes = parity_content_hashes(parity_output, parity_status)
    expected_parity_hashes = {
        "output_content_sha256": str(
            get_nested(config, "parent_anchor.parity_output_content_sha256")
        ),
        "status_content_sha256": str(
            get_nested(config, "parent_anchor.parity_status_content_sha256")
        ),
        "iteration_content_sha256": str(
            get_nested(config, "parent_anchor.parity_iteration_content_sha256")
        ),
    }
    sample_exact = sample_hashes == expected_sample_hashes
    parity_exact = parity_hashes == expected_parity_hashes
    return {
        "parent_sample_wells": sample_wells,
        "parent_parity_wells": parity_wells,
        "sample_series": len(status_subset),
        "parity_series": len(parity_status),
        "sample_hashes": sample_hashes,
        "expected_sample_hashes": expected_sample_hashes,
        "parity_hashes": parity_hashes,
        "expected_parity_hashes": expected_parity_hashes,
        "sample_exact_identity": sample_exact,
        "parity_exact_identity": parity_exact,
        "all_cross_run_parity_passed": sample_exact and parity_exact,
    }


def evaluate_full_gate(
    input_frame: pd.DataFrame,
    output_frame: pd.DataFrame,
    status: pd.DataFrame,
    *,
    raw_identity_sha256: str,
    parent_anchors_match: bool,
    cross_run_parity: Mapping[str, Any],
    audit_elapsed_seconds: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    expected_wells = int(get_nested(config, "audit.full.expected_wells"))
    expected_series = int(get_nested(config, "audit.full.expected_series"))
    runtime_limit = float(get_nested(config, "audit.full.runtime_limit_seconds"))
    expected_raw_sha = str(get_nested(config, "data.expected_raw_well_identity_sha256"))
    error_count = int(status["error"].astype(str).str.len().gt(0).sum()) if len(status) else 0
    criteria = {
        "parent_anchors_match": bool(parent_anchors_match),
        "raw_identity_match": raw_identity_sha256 == expected_raw_sha,
        "expected_well_coverage": int(status["well_id"].nunique()) == expected_wells,
        "expected_status_rows": len(status) == expected_series,
        "status_identity": not status.duplicated(
            ["branch", "well_id", "series_kind"]
        ).any(),
        "series_kind_coverage": set(status["series_kind"].astype(str))
        == set(SERIES_KINDS),
        "finite_input": bool(len(status) and status["finite_input"].astype(bool).all()),
        "finite_output": bool(
            len(status)
            and status["finite_output"].astype(bool).all()
            and np.isfinite(output_frame["output_gr"].to_numpy(np.float64)).all()
        ),
        "length_order_identity": bool(
            len(status)
            and status[["length_match", "order_match"]].astype(bool).all().all()
            and output_matches_input(input_frame, output_frame)
        ),
        "all_converged": bool(len(status) and status["converged"].astype(bool).all()),
        "silent_fallback_zero": bool(
            len(status) and not status["silent_fallback"].astype(bool).any()
        ),
        "error_count_zero": error_count == 0,
        "all_technical_pass": bool(
            len(status) and status["technical_pass"].astype(bool).all()
        ),
        "cross_run_parity_exact": bool(
            cross_run_parity.get("all_cross_run_parity_passed")
        ),
        "runtime_within_limit": audit_elapsed_seconds <= runtime_limit,
        "truth_or_scientific_score_loaded_false": True,
        "prediction_and_submission_absent": True,
    }
    full_pass = all(criteria.values())
    return {
        "experiment": EXPERIMENT_NAME,
        "branch": BRANCH_L1,
        "status": (
            "full_technical_pass"
            if full_pass
            else "full_technical_fail_closed"
        ),
        "full_technical_pass": full_pass,
        "criteria": criteria,
        "expected_wells": expected_wells,
        "actual_wells": int(status["well_id"].nunique()) if len(status) else 0,
        "expected_series": expected_series,
        "actual_series": len(status),
        "converged_series": int(status["converged"].astype(bool).sum()) if len(status) else 0,
        "technical_pass_series": int(
            status["technical_pass"].astype(bool).sum()
        ) if len(status) else 0,
        "silent_fallback_count": int(
            status["silent_fallback"].astype(bool).sum()
        ) if len(status) else 0,
        "error_count": error_count,
        "iterations": {
            "min": int(status["iterations"].min()) if len(status) else None,
            "mean": float(status["iterations"].mean()) if len(status) else None,
            "max": int(status["iterations"].max()) if len(status) else None,
        },
        "audit_elapsed_seconds": float(audit_elapsed_seconds),
        "runtime_limit_seconds": runtime_limit,
        "raw_well_identity_content_sha256": raw_identity_sha256,
        "input_content_sha256": dataframe_content_sha(input_frame),
        "output_content_sha256": dataframe_content_sha(output_frame),
        "status_content_sha256": dataframe_content_sha(status),
        "truth_or_scientific_score_loaded": False,
        "prediction": None,
        "submission": None,
        "failure_policy": "fail_closed_without_solver_rescue",
    }


# %% [markdown]
# ## 8. Full-audit orchestration and generated artifacts


# %%
def run_full_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    if not KAGGLE_WORKING_ROOT.exists() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError(
            "exp351 full audit must run on Kaggle; EXPERIMENT_ALLOW_LOCAL=1 is "
            "reserved for an explicitly approved local smoke run"
        )
    validate_technical_contract(config, require_run_approval=True)
    audit_started = time.perf_counter()
    outputs = artifact_dir()
    raw_dir = train_data_dir(config)

    parent_dir = resolve_parent_artifact_dir(config)
    parent_anchor_manifest, parent_sample = validate_parent_anchors(parent_dir, config)

    all_wells = enumerate_paired_wells(raw_dir)
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if len(all_wells) != expected_wells:
        raise ValueError(f"raw train has {len(all_wells)} wells, expected {expected_wells}")
    raw_identity = raw_well_identity_manifest(raw_dir, all_wells)
    raw_identity_sha = dataframe_content_sha(
        raw_identity,
        ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"],
    )
    expected_raw_sha = str(get_nested(config, "data.expected_raw_well_identity_sha256"))
    if raw_identity_sha != expected_raw_sha:
        raise ValueError(
            f"raw well identity SHA mismatch: {raw_identity_sha} != {expected_raw_sha}"
        )
    parent_sample_check = validate_parent_sample_against_raw(
        parent_sample, raw_identity, config
    )

    contract = build_technical_contract(config)
    contract_path = outputs / f"{OUTPUT_PREFIX}_scientific_contract.json"
    write_json(contract_path, contract)
    parent_anchor_manifest["raw_parent_sample_check"] = parent_sample_check
    parent_anchor_path = outputs / f"{OUTPUT_PREFIX}_parent_anchor_manifest.json"
    write_json(parent_anchor_path, parent_anchor_manifest)
    raw_identity_artifact = write_csv_plain(
        raw_identity,
        outputs / f"{OUTPUT_PREFIX}_raw_identity_manifest.csv",
    )

    preparation_started = time.perf_counter()
    prepared = load_prepared_full(raw_dir, all_wells, config)
    input_frame = build_input_frame(prepared)
    preparation_elapsed = float(time.perf_counter() - preparation_started)
    full_output, full_status, solver_elapsed = run_l1_full(prepared, config)
    del prepared

    cross_run_parity = evaluate_cross_run_parity(
        input_frame,
        full_output,
        full_status,
        parent_sample,
        config,
    )
    parity_path = outputs / f"{OUTPUT_PREFIX}_cross_run_parity.json"
    write_json(parity_path, cross_run_parity)

    input_artifact = write_csv_gzip(
        input_frame,
        outputs / f"{OUTPUT_PREFIX}_full_input.csv.gz",
    )
    output_artifact = write_csv_gzip(
        full_output,
        outputs / f"{OUTPUT_PREFIX}_full_output.csv.gz",
    )
    status_artifact = write_csv_gzip(
        full_status,
        outputs / f"{OUTPUT_PREFIX}_full_solver_status.csv.gz",
    )
    audit_elapsed = float(time.perf_counter() - audit_started)
    full_gate = evaluate_full_gate(
        input_frame,
        full_output,
        full_status,
        raw_identity_sha256=raw_identity_sha,
        parent_anchors_match=bool(
            parent_anchor_manifest.get("all_parent_anchors_match")
        ),
        cross_run_parity=cross_run_parity,
        audit_elapsed_seconds=audit_elapsed,
        config=config,
    )
    gate_path = outputs / f"{OUTPUT_PREFIX}_full_gate.json"
    write_json(gate_path, full_gate)

    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": full_gate["status"],
        "route": "pf_beam",
        "branch": BRANCH_L1,
        "wells": len(all_wells),
        "series_runs": len(full_status),
        "preparation_elapsed_seconds": preparation_elapsed,
        "solver_elapsed_seconds": solver_elapsed,
        "audit_elapsed_seconds": audit_elapsed,
        "full_technical_pass": full_gate["full_technical_pass"],
        "technical_eligibility_for_separate_scientific_experiment": full_gate[
            "full_technical_pass"
        ],
        "scientific_score": None,
        "cv": None,
        "prediction": None,
        "submission": None,
        "exp304_selected_denoiser_changed": False,
        "artifacts": {
            "contract": str(contract_path),
            "parent_anchor": str(parent_anchor_path),
            "raw_identity": raw_identity_artifact,
            "input": input_artifact,
            "output": output_artifact,
            "solver_status": status_artifact,
            "cross_run_parity": str(parity_path),
            "full_gate": str(gate_path),
        },
        "runtime_environment": runtime_manifest(),
        "failure_policy": "fail_closed_without_solver_rescue",
    }
    summary_path = outputs / f"{OUTPUT_PREFIX}_summary.json"
    write_json(summary_path, summary)
    write_json(metrics_output_path(), summary)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 9. Setup and configuration preview


# %%
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    validate_technical_contract(CONFIG)
    CONTRACT_PREVIEW = build_technical_contract(CONFIG)
    print(json.dumps(to_jsonable(CONTRACT_PREVIEW), indent=2, sort_keys=True))


# %% [markdown]
# ## 10. Run the separately approved Kaggle CPU full audit


# %%
if EXECUTE_NOTEBOOK:
    if not bool(get_nested(CONFIG, "execution.run_full_l1")):
        raise RuntimeError(
            "exp351 full audit is implemented, but Kaggle package/push/run is not "
            "approved; scientific scoring, inference, and submission remain disabled"
        )
    FULL_AUDIT_SUMMARY = run_full_experiment(CONFIG)
