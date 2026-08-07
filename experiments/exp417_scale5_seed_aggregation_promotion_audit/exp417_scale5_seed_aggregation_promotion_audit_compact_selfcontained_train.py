# %% [markdown]
# # exp417 scale-5 seed aggregation promotion audit train
#
# Saved-OOF, zero-PF promotion audit for one fixed readout change. The
# arithmetic mean and the full-suffix likelihood-temperature-5 weighted mean
# are loaded from the identical exp404 x1.0 particle/seed trajectory bank.
# Prediction identity and provenance are frozen before truth, fold, or
# hidden-like reporting roles are read.

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime, configuration, path, and SHA helpers
# 3. Frozen scientific contract and saved-input preflight
# 4. exp404 prediction identity and same-bank freeze
# 5. Late truth, fold, control, and hidden-like joins
# 6. Direct, scope, fold, blend, and well-tail metrics
# 7. Promotion gate
# 8. Generated artifacts and execution orchestration
# 9. Setup and configuration preview
# 10. Run the Kaggle CPU audit

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPERIMENT_NAME = "exp417_scale5_seed_aggregation_promotion_audit"
OUTPUT_PREFIX = EXPERIMENT_NAME
PRIMARY_CONTROL = "likpf_mean_x1p0"
PRIMARY_CANDIDATE = "likpf_scale_5_x1p0"
HMM_CONTROL = "exp209_hmm_tvt"
CONTROL_BLEND = "exp209_hmm_arithmetic_50_50"
CANDIDATE_BLEND = "exp209_hmm_scale5_50_50"
PARENT_PREDICTION_COLUMNS = (
    "likpf_scale_5_x1p0",
    "likpf_scale_5_x1p3",
    "likpf_mean_x1p0",
    "likpf_mean_x1p3",
)
PARENT_TABLE_COLUMNS = (
    "id",
    "well_id",
    "row_idx",
    "suffix_offset",
    "last_known_tvt",
    "md_since",
    "raw_gr_observed",
    "well_missing_fraction",
    *PARENT_PREDICTION_COLUMNS,
)
STAGE_A_LOGICAL_COLUMNS = (
    "id",
    "well_id",
    "row_idx",
    PRIMARY_CONTROL,
    PRIMARY_CANDIDATE,
)
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP417_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


# %% [markdown]
# ## 2. Runtime, configuration, path, and SHA helpers


# %%
def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
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
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


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
    raise FileNotFoundError(f"exp417 config not found; checked={checked}")


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


def inspect_gzip_csv(path: str | Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    newline_count = 0
    last_byte = b""
    with gzip.open(path, "rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(chunk)
            newline_count += chunk.count(b"\n")
            if chunk:
                last_byte = chunk[-1:]
    line_count = newline_count + int(bool(last_byte) and last_byte != b"\n")
    csv_path = Path(path)
    return {
        "path": str(csv_path),
        "bytes": csv_path.stat().st_size,
        "raw_sha256": sha256_path(csv_path),
        "decompressed_sha256": digest.hexdigest(),
        "content_sha256": digest.hexdigest(),
        "data_rows": max(0, line_count - 1),
        "columns": pd.read_csv(csv_path, nrows=0, compression="gzip").columns.astype(str).tolist(),
    }


def dataframe_content_sha(
    frame: pd.DataFrame,
    columns: list[str] | tuple[str, ...] | None = None,
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


def dataframe_schema_sha(frame: pd.DataFrame) -> str:
    return mapping_sha256({str(column): str(dtype) for column, dtype in frame.dtypes.items()})


def resolve_existing(filename: str, candidates: Iterable[str]) -> Path:
    root = project_root()
    checked: list[str] = []
    for raw in candidates:
        candidate = Path(str(raw))
        paths = (
            candidate if candidate.name == filename else candidate / filename,
            root / candidate if candidate.name == filename else root / candidate / filename,
            Path.cwd() / candidate
            if candidate.name == filename
            else Path.cwd() / candidate / filename,
        )
        for path in paths:
            checked.append(str(path))
            if path.exists() and path.is_file():
                return path
    if KAGGLE_INPUT_ROOT.exists():
        for path in sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}")):
            if path.is_file():
                return path
    raise FileNotFoundError(f"could not resolve {filename}; checked={checked}")


def runtime_versions() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyyaml": getattr(yaml, "__version__", "unknown"),
    }


def _input_spec(config: dict[str, Any], key: str) -> dict[str, Any]:
    value = get_nested(config, f"data.{key}") or {}
    if not isinstance(value, dict):
        raise ValueError(f"data.{key} must be a mapping")
    return value


# %% [markdown]
# ## 3. Frozen scientific contract and saved-input preflight


# %%
def build_scientific_contract(config: dict[str, Any]) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "artifact_parent": get_nested(config, "lineage.parent"),
        "scientific_control": get_nested(config, "lineage.scientific_control"),
        "cause_evidence": get_nested(config, "lineage.evidence"),
        "truth_attached": False,
        "primary_control": PRIMARY_CONTROL,
        "primary_candidate": PRIMARY_CANDIDATE,
        "fixed_hmm_control": HMM_CONTROL,
        "aggregation": get_nested(config, "model.aggregation"),
        "pf_source_contract": get_nested(config, "model.pf_source_contract"),
        "execution_counts": get_nested(config, "model.execution_count"),
        "truth_freeze_policy": get_nested(config, "validation.truth_attachment"),
        "source_sha_contract": {
            "exp404": get_nested(config, "data.exp404_frozen_predictions"),
            "exp072": get_nested(config, "data.exp072_control.expected_decompressed_sha256"),
            "exp209_hmm": get_nested(
                config, "data.exp209_hmm_control.expected_decompressed_sha256"
            ),
            "fold": get_nested(config, "data.fold_assignment.expected_decompressed_sha256"),
            "hidden_like": get_nested(config, "data.hidden_like_assignment.expected_sha256"),
        },
        "controls": {
            "exp404_arithmetic": "saved_load_only",
            "exp404_scale5": "saved_load_only",
            "exp072_mean": "saved_load_only_technical_parity",
            "exp209_hmm": "saved_load_only_fixed_blend",
            "pf_reruns": 0,
            "hmm_reruns": 0,
            "model_reruns": 0,
        },
        "forbidden": get_nested(config, "guards.forbidden"),
    }
    contract["scientific_contract_sha256"] = mapping_sha256(contract)
    return contract


def validate_scientific_contract(
    config: dict[str, Any],
    *,
    require_run_approval: bool = False,
) -> dict[str, Any]:
    expected: dict[str, Any] = {
        "experiment.name": EXPERIMENT_NAME,
        "experiment.route": "pf_beam",
        "lineage.parent": "exp404_scale5_sigma_gr_likelihood_pf_ablation",
        "lineage.scientific_control": "exp072_exp063_full_replay_feature_cache",
        "implementation.enabled": True,
        "implementation.scope": "train_side_saved_oof_promotion_audit_only",
        "model.estimator": "none_saved_prediction_audit",
        "model.fitted_model": False,
        "model.pf_source_contract.particles": 500,
        "model.pf_source_contract.seeds": 128,
        "model.pf_source_contract.gr_scale_multiplier": 1.0,
        "model.aggregation.control.name": "arithmetic_seed_mean",
        "model.aggregation.candidate.name": (
            "full_suffix_likelihood_temperature_5_weighted_seed_mean"
        ),
        "model.aggregation.candidate.temperature": 5.0,
        "model.aggregation.candidate.target_free": True,
        "model.aggregation.candidate.causal_online": False,
        "model.aggregation.candidate.batch_inference_compatible": True,
        "model.aggregation.other_aggregations_enabled": False,
        "model.execution_count.saved_candidate_readouts": 1,
        "model.execution_count.scientific_candidates": 1,
        "model.execution_count.pf_well_runs": 0,
        "model.execution_count.parent_pf_control_reruns": 0,
        "model.execution_count.model_configs": 0,
        "model.execution_count.trained_folds": 0,
        "model.execution_count.boosters": 0,
        "model.execution_count.hmm_well_runs": 0,
        "model.execution_count.beam_well_runs": 0,
        "model.execution_count.gpu_runs": 0,
        "model.execution_count.reporting_folds": 5,
        "runtime.device": "cpu",
        "runtime.kaggle.enable_gpu": False,
        "runtime.kaggle.enable_internet": False,
        "execution.implementation_approved": True,
        "inference.mode": "disabled_until_stage_a_all_guards_pass_and_separate_approval",
        "inference.selected_candidate": None,
    }
    for key, value in expected.items():
        if get_nested(config, key) != value:
            raise ValueError(f"exp417 fixed contract mismatch: {key} must be {value!r}")
    if get_nested(config, "validation.primary_control") != PRIMARY_CONTROL:
        raise ValueError("exp417 primary control changed")
    if get_nested(config, "validation.primary_candidate") != PRIMARY_CANDIDATE:
        raise ValueError("exp417 primary candidate changed")
    hidden = _input_spec(config, "hidden_like_assignment")
    expected_roles = {
        "hidden_like_spatial": {"train": 573, "valid": 200},
        "hidden_like_typewell_purged": {
            "train": 557,
            "valid": 200,
            "purged_train_excluded": 16,
        },
    }
    if hidden.get("expected_role_counts") != expected_roles:
        raise ValueError("exp417 hidden-like expected_role_counts changed")
    if require_run_approval and not (
        bool(get_nested(config, "execution.canonical_notebook_adoption_approved"))
        and bool(get_nested(config, "execution.kaggle_package_approved"))
        and bool(get_nested(config, "execution.kaggle_push_approved"))
        and bool(get_nested(config, "execution.stage_a_run_approved"))
        and bool(get_nested(config, "execution.run_stage_a"))
    ):
        raise RuntimeError("exp417 canonical Notebook/package/push/Stage A run is not approved")
    return build_scientific_contract(config)


def validate_parent_scientific_contract(
    payload: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    expected_sha = str(
        get_nested(config, "data.exp404_frozen_predictions.expected_scientific_contract_sha256")
    )
    recorded_sha = str(payload.get("scientific_contract_sha256") or "")
    unhashed = dict(payload)
    unhashed.pop("scientific_contract_sha256", None)
    recomputed_sha = mapping_sha256(unhashed)
    required = {
        "experiment": "exp404_scale5_sigma_gr_likelihood_pf_ablation",
        "primary_control": "likpf_scale_5_x1p0",
        "primary_candidate": "likpf_scale_5_x1p3",
        "parity_columns": ["likpf_mean_x1p0", "likpf_mean_x1p3"],
        "pf.particles": 500,
        "pf.seeds": 128,
        "pf.primary_seed_weighting_scale": 5.0,
        "pf.arithmetic_mean_enabled_for_parity_only": True,
        "gr_scale.variants.gs_x1p0.multiplier": 1.0,
    }
    mismatches = {
        key: {"expected": value, "actual": get_nested(payload, key)}
        for key, value in required.items()
        if get_nested(payload, key) != value
    }
    passed = bool(
        not mismatches and recorded_sha == expected_sha and recomputed_sha == expected_sha
    )
    if not passed:
        raise ValueError(
            "exp404 scientific contract mismatch: "
            f"recorded={recorded_sha}, recomputed={recomputed_sha}, mismatches={mismatches}"
        )
    return {
        "passed": True,
        "recorded_sha256": recorded_sha,
        "recomputed_sha256": recomputed_sha,
        "same_x1p0_pf_call_readouts": [
            "likpf_mean_x1p0",
            "likpf_scale_5_x1p0",
        ],
        "temperature": 5.0,
        "particles": 500,
        "seeds": 128,
    }


def preflight_saved_inputs(config: dict[str, Any]) -> dict[str, Any]:
    prediction_spec = _input_spec(config, "exp404_frozen_predictions")
    specs = {
        "exp072_control": _input_spec(config, "exp072_control"),
        "exp209_hmm_control": _input_spec(config, "exp209_hmm_control"),
        "fold_assignment": _input_spec(config, "fold_assignment"),
        "hidden_like_assignment": _input_spec(config, "hidden_like_assignment"),
    }
    prediction_candidates = [str(value) for value in prediction_spec.get("candidates", [])]
    paths = {
        "exp404_predictions": resolve_existing(
            str(prediction_spec["filename"]),
            prediction_candidates,
        ),
        "exp404_audit": resolve_existing(
            str(prediction_spec["audit_filename"]),
            prediction_candidates,
        ),
        "exp404_contract": resolve_existing(
            str(prediction_spec["contract_filename"]),
            prediction_candidates,
        ),
    }
    for name, spec in specs.items():
        paths[name] = resolve_existing(
            str(spec["filename"]),
            [str(value) for value in spec.get("candidates", [])],
        )

    reports: dict[str, Any] = {}
    reports["exp404_predictions"] = inspect_gzip_csv(paths["exp404_predictions"])
    if reports["exp404_predictions"]["raw_sha256"] != str(prediction_spec["expected_raw_sha256"]):
        raise ValueError("exp404 prediction raw gzip SHA mismatch")
    if reports["exp404_predictions"]["decompressed_sha256"] != str(
        prediction_spec["expected_decompressed_sha256"]
    ):
        raise ValueError("exp404 prediction decompressed SHA mismatch")

    for name in ("exp072_control", "exp209_hmm_control", "fold_assignment"):
        reports[name] = inspect_gzip_csv(paths[name])
        if reports[name]["decompressed_sha256"] != str(specs[name]["expected_decompressed_sha256"]):
            raise ValueError(f"{name} decompressed SHA mismatch")

    audit_sha = sha256_path(paths["exp404_audit"])
    if audit_sha != str(prediction_spec["expected_audit_raw_sha256"]):
        raise ValueError("exp404 well audit raw SHA mismatch")
    reports["exp404_audit"] = {
        "path": str(paths["exp404_audit"]),
        "bytes": paths["exp404_audit"].stat().st_size,
        "raw_sha256": audit_sha,
        "columns": pd.read_csv(paths["exp404_audit"], nrows=0).columns.astype(str).tolist(),
    }

    contract_payload = json.loads(paths["exp404_contract"].read_text())
    parent_contract = validate_parent_scientific_contract(contract_payload, config)
    reports["exp404_contract"] = {
        "path": str(paths["exp404_contract"]),
        "bytes": paths["exp404_contract"].stat().st_size,
        "raw_sha256": sha256_path(paths["exp404_contract"]),
        "scientific_contract_sha256": parent_contract["recorded_sha256"],
    }

    hidden_sha = sha256_path(paths["hidden_like_assignment"])
    if hidden_sha != str(specs["hidden_like_assignment"]["expected_sha256"]):
        raise ValueError("hidden-like assignment raw SHA mismatch")
    reports["hidden_like_assignment"] = {
        "path": str(paths["hidden_like_assignment"]),
        "bytes": paths["hidden_like_assignment"].stat().st_size,
        "raw_sha256": hidden_sha,
        "columns": pd.read_csv(paths["hidden_like_assignment"], nrows=0)
        .columns.astype(str)
        .tolist(),
    }

    required_columns = {
        "exp404_predictions": set(PARENT_TABLE_COLUMNS),
        "exp404_audit": {
            "well_id",
            "status",
            "gs_base",
            "gs_x1p0",
            "seed_base",
            "seed_base_x1p0",
            "pf_well_runs",
            "seeds",
            "particles",
        },
        "exp072_control": {
            "id",
            "well",
            str(specs["exp072_control"]["anchor_column"]),
            str(specs["exp072_control"]["delta_column"]),
        },
        "exp209_hmm_control": {
            "id",
            "well",
            str(specs["exp209_hmm_control"]["prediction_column"]),
        },
        "fold_assignment": {
            *[str(value) for value in specs["fold_assignment"]["safe_columns"]],
            *[str(value) for value in specs["fold_assignment"]["truth_columns"]],
        },
        "hidden_like_assignment": {
            "well_id",
            *[str(value) for value in specs["hidden_like_assignment"]["role_columns"].values()],
        },
    }
    for name, required in required_columns.items():
        missing = sorted(required - set(reports[name]["columns"]))
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}")
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    for name in (
        "exp404_predictions",
        "exp072_control",
        "exp209_hmm_control",
        "fold_assignment",
    ):
        if int(reports[name]["data_rows"]) != expected_rows:
            raise ValueError(f"{name} row count mismatch")
    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "reports": reports,
        "parent_contract": parent_contract,
        "all_input_sha_matches": True,
        "truth_or_reporting_values_parsed_before_freeze": {
            "unknown_suffix_tvt_rows": 0,
            "error_rows": 0,
            "fold_rows": 0,
            "hidden_like_role_rows": 0,
        },
    }


# %% [markdown]
# ## 4. exp404 prediction identity and same-bank freeze


# %%
@dataclass
class TruthAccessLedger:
    prediction_frozen: bool = False
    unknown_suffix_tvt_rows_before_freeze: int = 0
    error_rows_before_freeze: int = 0
    fold_rows_before_freeze: int = 0
    hidden_like_role_rows_before_freeze: int = 0
    unknown_suffix_tvt_rows_after_freeze: int = 0
    error_rows_after_freeze: int = 0
    fold_rows_after_freeze: int = 0
    hidden_like_role_rows_after_freeze: int = 0

    def mark_frozen(self) -> None:
        if self.prediction_frozen:
            raise RuntimeError("prediction identity is already frozen")
        before = (
            self.unknown_suffix_tvt_rows_before_freeze,
            self.error_rows_before_freeze,
            self.fold_rows_before_freeze,
            self.hidden_like_role_rows_before_freeze,
        )
        if any(value != 0 for value in before):
            raise RuntimeError("truth/reporting values were read before prediction freeze")
        self.prediction_frozen = True

    def require_frozen(self) -> None:
        if not self.prediction_frozen:
            raise RuntimeError("late attachment requires a frozen prediction identity")

    def report(self) -> dict[str, Any]:
        return {
            "prediction_frozen": self.prediction_frozen,
            "before_freeze": {
                "unknown_suffix_tvt_rows": self.unknown_suffix_tvt_rows_before_freeze,
                "error_rows": self.error_rows_before_freeze,
                "fold_rows": self.fold_rows_before_freeze,
                "hidden_like_role_rows": self.hidden_like_role_rows_before_freeze,
            },
            "after_freeze": {
                "unknown_suffix_tvt_rows": self.unknown_suffix_tvt_rows_after_freeze,
                "error_rows": self.error_rows_after_freeze,
                "fold_rows": self.fold_rows_after_freeze,
                "hidden_like_role_rows": self.hidden_like_role_rows_after_freeze,
            },
        }


def load_and_freeze_prediction_identity(
    preflight: dict[str, Any],
    config: dict[str, Any],
    ledger: TruthAccessLedger,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if ledger.prediction_frozen:
        raise RuntimeError("prediction identity is already frozen")
    prediction_spec = _input_spec(config, "exp404_frozen_predictions")
    prediction = pd.read_csv(
        preflight["paths"]["exp404_predictions"],
        dtype={"id": str, "well_id": str},
        compression="gzip",
    )
    prediction["id"] = prediction["id"].astype(object)
    prediction["well_id"] = prediction["well_id"].astype(object)
    for column in ("row_idx", "suffix_offset"):
        prediction[column] = pd.to_numeric(prediction[column], errors="raise").astype(np.int64)
    prediction["raw_gr_observed"] = prediction["raw_gr_observed"].astype(bool)
    for column in (
        "last_known_tvt",
        "md_since",
        "well_missing_fraction",
        *PARENT_PREDICTION_COLUMNS,
    ):
        prediction[column] = pd.to_numeric(prediction[column], errors="raise").astype(np.float64)
    if prediction.columns.astype(str).tolist() != list(PARENT_TABLE_COLUMNS):
        raise ValueError("exp404 prediction columns or order mismatch")
    forbidden = {str(value) for value in prediction_spec.get("forbidden_columns", [])}
    if forbidden & set(prediction.columns.astype(str)):
        raise ValueError("exp404 prediction contains forbidden late-readout columns")
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    if (
        len(prediction) != expected_rows
        or prediction["well_id"].nunique() != expected_wells
        or prediction["id"].duplicated().any()
        or prediction.duplicated(["well_id", "row_idx"]).any()
    ):
        raise ValueError("exp404 prediction identity or coverage mismatch")
    if not np.isfinite(prediction[[PRIMARY_CONTROL, PRIMARY_CANDIDATE]].to_numpy(np.float64)).all():
        raise ValueError("exp404 Stage A readouts contain non-finite values")

    parent_logical_columns = [
        "id",
        "well_id",
        "row_idx",
        *PARENT_PREDICTION_COLUMNS,
    ]
    parent_logical_sha = dataframe_content_sha(prediction, parent_logical_columns)
    if parent_logical_sha != str(prediction_spec["expected_logical_sha256"]):
        raise ValueError("exp404 prediction logical content SHA mismatch")
    schema_sha = dataframe_schema_sha(prediction)
    if schema_sha != str(prediction_spec["expected_schema_sha256"]):
        raise ValueError("exp404 prediction schema SHA mismatch")
    stage_a_logical_sha = dataframe_content_sha(prediction, STAGE_A_LOGICAL_COLUMNS)

    audit = pd.read_csv(
        preflight["paths"]["exp404_audit"],
        dtype={"well_id": str, "status": str},
    )
    required_audit_columns = {
        "well_id",
        "status",
        "gs_base",
        "gs_x1p0",
        "seed_base",
        "seed_base_x1p0",
        "pf_well_runs",
        "seeds",
        "particles",
    }
    missing_audit = sorted(required_audit_columns - set(audit.columns.astype(str)))
    if missing_audit:
        raise ValueError(f"exp404 well audit missing columns: {missing_audit}")
    for column in (
        "gs_base",
        "gs_x1p0",
        "seed_base",
        "seed_base_x1p0",
        "pf_well_runs",
        "seeds",
        "particles",
    ):
        audit[column] = pd.to_numeric(audit[column], errors="raise")
    same_seed_labels = np.array_equal(
        audit["seed_base"].to_numpy(np.int64),
        audit["seed_base_x1p0"].to_numpy(np.int64),
    )
    same_x1p0_scale = np.array_equal(
        audit["gs_base"].to_numpy(np.float64),
        audit["gs_x1p0"].to_numpy(np.float64),
    )
    same_bank_evidence = {
        "passed": bool(
            len(audit) == expected_wells
            and audit["well_id"].nunique() == expected_wells
            and audit["status"].eq("ok").all()
            and same_seed_labels
            and same_x1p0_scale
            and audit["seeds"].eq(128).all()
            and audit["particles"].eq(500).all()
            and int(audit["pf_well_runs"].sum()) == expected_wells * 2
            and bool(preflight["parent_contract"]["passed"])
        ),
        "audit_wells": len(audit),
        "all_source_wells_ok": bool(audit["status"].eq("ok").all()),
        "same_seed_labels": same_seed_labels,
        "same_x1p0_gr_scale": same_x1p0_scale,
        "particles": sorted(audit["particles"].astype(int).unique().tolist()),
        "seeds": sorted(audit["seeds"].astype(int).unique().tolist()),
        "source_pf_well_runs": int(audit["pf_well_runs"].sum()),
        "stage_a_pf_well_runs": 0,
        "parent_contract": preflight["parent_contract"],
    }
    if not same_bank_evidence["passed"]:
        raise ValueError(f"exp404 same-bank provenance mismatch: {same_bank_evidence}")

    frozen = {
        "frozen_before_truth_attachment": True,
        "rows": len(prediction),
        "wells": int(prediction["well_id"].nunique()),
        "control_column": PRIMARY_CONTROL,
        "candidate_column": PRIMARY_CANDIDATE,
        "parent_logical_columns": parent_logical_columns,
        "parent_logical_content_sha256": parent_logical_sha,
        "stage_a_logical_columns": list(STAGE_A_LOGICAL_COLUMNS),
        "stage_a_logical_content_sha256": stage_a_logical_sha,
        "schema_sha256": schema_sha,
        "raw_gzip_sha256": preflight["reports"]["exp404_predictions"]["raw_sha256"],
        "decompressed_content_sha256": preflight["reports"]["exp404_predictions"][
            "decompressed_sha256"
        ],
        "exp404_well_audit_raw_sha256": preflight["reports"]["exp404_audit"]["raw_sha256"],
        "exp404_scientific_contract_sha256": preflight["parent_contract"]["recorded_sha256"],
        "same_bank_evidence": same_bank_evidence,
        "truth_or_reporting_values_parsed_before_freeze": ledger.report()["before_freeze"],
        "stage_a_execution_counts": get_nested(config, "model.execution_count"),
    }
    ledger.mark_frozen()
    return prediction, audit, frozen


def _require_frozen_prediction(frozen: dict[str, Any]) -> None:
    if not bool(frozen.get("frozen_before_truth_attachment")):
        raise RuntimeError("late attachment requires a frozen prediction identity")
    for key in (
        "parent_logical_content_sha256",
        "stage_a_logical_content_sha256",
        "schema_sha256",
        "raw_gzip_sha256",
        "decompressed_content_sha256",
    ):
        if len(str(frozen.get(key) or "")) != 64:
            raise RuntimeError(f"frozen prediction identity is missing {key}")


# %% [markdown]
# ## 5. Late truth, fold, control, and hidden-like joins


# %%
def _align_on_id(
    frame: pd.DataFrame,
    source: pd.DataFrame,
    columns: list[str],
    *,
    label: str,
) -> pd.DataFrame:
    if source["id"].astype(str).duplicated().any():
        raise ValueError(f"{label} contains duplicate IDs")
    lookup = source.assign(id=source["id"].astype(str)).set_index("id")
    aligned = lookup.reindex(frame["id"].astype(str))
    if aligned[columns].isna().any().any():
        raise ValueError(f"{label} has missing aligned rows")
    result = frame.copy()
    for column in columns:
        result[column] = aligned[column].to_numpy()
    return result


def materialize_saved_exp072_mean(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> np.ndarray:
    spec = _input_spec(config, "exp072_control")
    anchor = pd.to_numeric(frame[str(spec["anchor_column"])], errors="raise").to_numpy(np.float64)
    delta = pd.to_numeric(frame[str(spec["delta_column"])], errors="raise").to_numpy(np.float64)
    values = anchor + delta
    if not np.isfinite(values).all():
        raise ValueError("saved exp072 mean materialization produced non-finite values")
    return values


def load_late_readout_frame(
    prediction: pd.DataFrame,
    frozen: dict[str, Any],
    preflight: dict[str, Any],
    config: dict[str, Any],
    ledger: TruthAccessLedger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require_frozen_prediction(frozen)
    ledger.require_frozen()
    parent_sha = dataframe_content_sha(
        prediction,
        list(frozen["parent_logical_columns"]),
    )
    stage_a_sha = dataframe_content_sha(
        prediction,
        list(frozen["stage_a_logical_columns"]),
    )
    if parent_sha != str(frozen["parent_logical_content_sha256"]):
        raise ValueError("parent prediction changed after identity freeze")
    if stage_a_sha != str(frozen["stage_a_logical_content_sha256"]):
        raise ValueError("Stage A readouts changed after identity freeze")

    fold_spec = _input_spec(config, "fold_assignment")
    safe_columns = [str(value) for value in fold_spec["safe_columns"]]
    truth_columns = [str(value) for value in fold_spec["truth_columns"]]
    if set(safe_columns) != {"well_id", "row_idx", "suffix_offset", "fold"}:
        raise ValueError("exp417 fold allowlist must contain identity/fold columns only")
    if set(safe_columns) & {str(value) for value in fold_spec.get("forbidden_decoder_columns", [])}:
        raise ValueError("exp417 fold allowlist contains forbidden decoder columns")
    fold = pd.read_csv(
        preflight["paths"]["fold_assignment"],
        usecols=[*safe_columns, *[value for value in truth_columns if value not in safe_columns]],
        dtype={"well_id": str},
    )
    for column in ("row_idx", "suffix_offset", "fold"):
        fold[column] = pd.to_numeric(fold[column], errors="raise").astype(np.int64)
    fold["tvt_true"] = pd.to_numeric(fold["tvt_true"], errors="raise").astype(np.float64)
    if fold.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("late fold/truth identity is duplicated")
    ledger.fold_rows_after_freeze += len(fold)
    ledger.unknown_suffix_tvt_rows_after_freeze += len(fold)
    frame = prediction[
        [
            "id",
            "well_id",
            "row_idx",
            "suffix_offset",
            "md_since",
            "raw_gr_observed",
            "well_missing_fraction",
            PRIMARY_CONTROL,
            PRIMARY_CANDIDATE,
        ]
    ].copy()
    frame = frame.merge(
        fold,
        on=["well_id", "row_idx"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_exp226"),
        sort=False,
    )
    if frame[["fold", "tvt_true", "suffix_offset_exp226"]].isna().any().any():
        raise ValueError("late fold/truth attachment is incomplete")
    if not np.array_equal(
        frame["suffix_offset"].to_numpy(np.int64),
        frame["suffix_offset_exp226"].to_numpy(np.int64),
    ):
        raise ValueError("exp226 suffix offset identity mismatch")
    frame = frame.drop(columns=["suffix_offset_exp226"]).rename(columns={"tvt_true": "true_tvt"})

    exp072_spec = _input_spec(config, "exp072_control")
    exp072_columns = [
        "id",
        str(exp072_spec["anchor_column"]),
        str(exp072_spec["delta_column"]),
    ]
    exp072 = pd.read_csv(
        preflight["paths"]["exp072_control"],
        usecols=exp072_columns,
        dtype={"id": str},
    )
    exp072["saved_exp072_likpf_mean"] = materialize_saved_exp072_mean(exp072, config)
    frame = _align_on_id(
        frame,
        exp072[["id", "saved_exp072_likpf_mean"]],
        ["saved_exp072_likpf_mean"],
        label="saved exp072 arithmetic mean",
    )

    hmm_spec = _input_spec(config, "exp209_hmm_control")
    hmm_column = str(hmm_spec["prediction_column"])
    hmm = pd.read_csv(
        preflight["paths"]["exp209_hmm_control"],
        usecols=["id", hmm_column],
        dtype={"id": str},
    ).rename(columns={hmm_column: HMM_CONTROL})
    hmm[HMM_CONTROL] = pd.to_numeric(hmm[HMM_CONTROL], errors="raise").astype(np.float64)
    frame = _align_on_id(
        frame,
        hmm,
        [HMM_CONTROL],
        label="saved exp209 exact HMM",
    )

    hidden_spec = _input_spec(config, "hidden_like_assignment")
    role_columns = {
        str(scope): str(column) for scope, column in hidden_spec["role_columns"].items()
    }
    hidden = pd.read_csv(
        preflight["paths"]["hidden_like_assignment"],
        usecols=["well_id", *role_columns.values()],
        dtype={"well_id": str},
    )
    if hidden["well_id"].duplicated().any():
        raise ValueError("hidden-like assignment contains duplicate wells")
    ledger.hidden_like_role_rows_after_freeze += len(hidden)
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
            for key, value in hidden_spec["expected_role_counts"][scope].items()
        }
        if actual != expected:
            raise ValueError(f"hidden-like role counts mismatch for {scope}")
    frame = frame.merge(hidden, on="well_id", how="left", validate="many_to_one")
    if frame[list(role_columns.values())].isna().any().any():
        raise ValueError("hidden-like role attachment is incomplete")
    frame["hidden_like_spatial"] = frame[role_columns["hidden_like_spatial"]].eq("valid")
    frame["hidden_like_typewell_purged"] = frame[role_columns["hidden_like_typewell_purged"]].eq(
        "valid"
    )

    frame[CONTROL_BLEND] = 0.5 * frame[HMM_CONTROL] + 0.5 * frame[PRIMARY_CONTROL]
    frame[CANDIDATE_BLEND] = 0.5 * frame[HMM_CONTROL] + 0.5 * frame[PRIMARY_CANDIDATE]
    finite_columns = [
        "true_tvt",
        "saved_exp072_likpf_mean",
        HMM_CONTROL,
        PRIMARY_CONTROL,
        PRIMARY_CANDIDATE,
        CONTROL_BLEND,
        CANDIDATE_BLEND,
    ]
    if not np.isfinite(frame[finite_columns].to_numpy(np.float64)).all():
        raise ValueError("late readout contains non-finite values")
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    if sorted(frame["fold"].astype(int).unique().tolist()) != expected_folds:
        raise ValueError("reporting fold set mismatch")
    return frame, {
        "truth_attached_after_prediction_freeze": True,
        "parent_prediction_sha256_reverified": parent_sha,
        "stage_a_prediction_sha256_reverified": stage_a_sha,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "folds": expected_folds,
        "truth_access_ledger": ledger.report(),
    }


# %% [markdown]
# ## 6. Direct, scope, fold, blend, and well-tail metrics


# %%
def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def metric_scopes(frame: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
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


def paired_metric_row(
    frame: pd.DataFrame,
    mask: np.ndarray,
    *,
    comparison: str,
    candidate_column: str,
    control_column: str,
    scope: str,
) -> dict[str, Any]:
    if not bool(mask.any()):
        raise ValueError(f"metric scope {scope} is empty")
    selected = frame.loc[mask]
    truth = selected["true_tvt"].to_numpy(np.float64)
    candidate = selected[candidate_column].to_numpy(np.float64)
    control = selected[control_column].to_numpy(np.float64)
    candidate_rmse = rmse(truth, candidate)
    control_rmse = rmse(truth, control)
    return {
        "comparison": comparison,
        "scope": scope,
        "rows": len(selected),
        "wells": int(selected["well_id"].nunique()),
        "candidate_column": candidate_column,
        "control_column": control_column,
        "candidate_rmse": candidate_rmse,
        "control_rmse": control_rmse,
        "candidate_mae": float(np.mean(np.abs(candidate - truth))),
        "control_mae": float(np.mean(np.abs(control - truth))),
        "candidate_bias": float(np.mean(candidate - truth)),
        "control_bias": float(np.mean(control - truth)),
        "candidate_within_10ft": float(np.mean(np.abs(candidate - truth) <= 10.0)),
        "control_within_10ft": float(np.mean(np.abs(control - truth) <= 10.0)),
        "improvement_ft": control_rmse - candidate_rmse,
        "delta_rmse_candidate_minus_control": candidate_rmse - control_rmse,
    }


def build_metric_outputs(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparisons = {
        "direct_scale5_vs_arithmetic": (PRIMARY_CANDIDATE, PRIMARY_CONTROL),
        "fixed_exp209_hmm_likpf_50_50": (CANDIDATE_BLEND, CONTROL_BLEND),
    }
    paired_rows: list[dict[str, Any]] = []
    by_well_rows: list[dict[str, Any]] = []
    for comparison, (candidate_column, control_column) in comparisons.items():
        for scope, mask in metric_scopes(frame):
            paired_rows.append(
                paired_metric_row(
                    frame,
                    mask,
                    comparison=comparison,
                    candidate_column=candidate_column,
                    control_column=control_column,
                    scope=scope,
                )
            )
        for well, group in frame.groupby("well_id", sort=True):
            truth = group["true_tvt"].to_numpy(np.float64)
            candidate_rmse = rmse(
                truth,
                group[candidate_column].to_numpy(np.float64),
            )
            control_rmse = rmse(
                truth,
                group[control_column].to_numpy(np.float64),
            )
            by_well_rows.append(
                {
                    "comparison": comparison,
                    "well_id": str(well),
                    "rows": len(group),
                    "candidate_rmse": candidate_rmse,
                    "control_rmse": control_rmse,
                    "improvement_ft": control_rmse - candidate_rmse,
                    "delta_rmse_candidate_minus_control": candidate_rmse - control_rmse,
                    "well_missing_fraction": float(group["well_missing_fraction"].iloc[0]),
                }
            )
    return pd.DataFrame(paired_rows), pd.DataFrame(by_well_rows)


def build_parity_metrics(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    truth = frame["true_tvt"].to_numpy(np.float64)
    actual = {
        "saved_exp072_mean": rmse(
            truth,
            frame["saved_exp072_likpf_mean"].to_numpy(np.float64),
        ),
        "exp404_arithmetic_x1p0": rmse(
            truth,
            frame[PRIMARY_CONTROL].to_numpy(np.float64),
        ),
        "saved_exp209_hmm": rmse(
            truth,
            frame[HMM_CONTROL].to_numpy(np.float64),
        ),
        "exp209_hmm_exp404_arithmetic_50_50": rmse(
            truth,
            frame[CONTROL_BLEND].to_numpy(np.float64),
        ),
    }
    tolerance = float(
        get_nested(config, "guards.technical.require_control_rmse_parity_vs_exp072_atol_ft")
    )
    exp072_expected = float(get_nested(config, "data.exp072_control.expected_rmse_ft"))
    checks = {
        "saved_exp072_mean_reference": {
            "actual": actual["saved_exp072_mean"],
            "expected": exp072_expected,
            "tolerance_ft": tolerance,
        },
        "exp404_arithmetic_vs_exp072_reference": {
            "actual": actual["exp404_arithmetic_x1p0"],
            "expected": exp072_expected,
            "tolerance_ft": tolerance,
        },
        "exp404_arithmetic_vs_saved_exp072": {
            "actual": actual["exp404_arithmetic_x1p0"],
            "expected": actual["saved_exp072_mean"],
            "tolerance_ft": tolerance,
        },
        "saved_exp209_hmm_reference": {
            "actual": actual["saved_exp209_hmm"],
            "expected": float(get_nested(config, "data.exp209_hmm_control.expected_rmse_ft")),
            "tolerance_ft": tolerance,
        },
        "fixed_exp209_hmm_arithmetic_50_50_reference": {
            "actual": actual["exp209_hmm_exp404_arithmetic_50_50"],
            "expected": float(
                get_nested(
                    config,
                    "data.exp209_hmm_control.expected_fixed_hmm_likpf_50_50_rmse_ft",
                )
            ),
            "tolerance_ft": tolerance,
        },
    }
    for check in checks.values():
        check["absolute_difference"] = abs(check["actual"] - check["expected"])
        check["passed"] = bool(check["absolute_difference"] <= check["tolerance_ft"])
    return {
        "experiment": EXPERIMENT_NAME,
        "policy": "saved_control_technical_parity_only",
        "actual_rmse": actual,
        "checks": checks,
        "passed": bool(all(check["passed"] for check in checks.values())),
    }


def _metric_row(
    paired_metrics: pd.DataFrame,
    comparison: str,
    scope: str,
) -> pd.Series:
    selected = paired_metrics.loc[
        paired_metrics["comparison"].eq(comparison) & paired_metrics["scope"].eq(scope)
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one metric row for comparison={comparison}, scope={scope}")
    return selected.iloc[0]


# %% [markdown]
# ## 7. Promotion gate


# %%
def evaluate_promotion_gate(
    frame: pd.DataFrame,
    paired_metrics: pd.DataFrame,
    by_well_metrics: pd.DataFrame,
    parity_metrics: dict[str, Any],
    preflight: dict[str, Any],
    frozen: dict[str, Any],
    ledger: TruthAccessLedger,
    runtime_seconds: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    technical_config = get_nested(config, "guards.technical") or {}
    scientific_config = get_nested(config, "guards.scientific") or {}
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    before_freeze = ledger.report()["before_freeze"]
    execution_counts = get_nested(config, "model.execution_count") or {}
    zero_execution_keys = (
        "pf_well_runs",
        "parent_pf_control_reruns",
        "model_configs",
        "trained_folds",
        "boosters",
        "hmm_well_runs",
        "beam_well_runs",
        "gpu_runs",
    )
    finite_coverage = float(
        np.isfinite(
            frame[
                [
                    PRIMARY_CONTROL,
                    PRIMARY_CANDIDATE,
                    HMM_CONTROL,
                    CONTROL_BLEND,
                    CANDIDATE_BLEND,
                ]
            ].to_numpy(np.float64)
        ).mean()
    )
    technical = {
        "all_input_sha_matches": bool(preflight["all_input_sha_matches"]),
        "prediction_rows": len(frame),
        "prediction_wells": int(frame["well_id"].nunique()),
        "reporting_folds": sorted(frame["fold"].astype(int).unique().tolist()),
        "finite_prediction_coverage": finite_coverage,
        "same_x1p0_seed_bank": frozen["same_bank_evidence"],
        "temperature": float(get_nested(config, "model.aggregation.candidate.temperature")),
        "truth_or_reporting_values_parsed_before_freeze": before_freeze,
        "stage_a_execution_counts": execution_counts,
        "all_stage_a_generation_and_training_counts_zero": bool(
            all(int(execution_counts[key]) == 0 for key in zero_execution_keys)
        ),
        "parity_metrics": parity_metrics,
        "runtime_seconds": runtime_seconds,
        "runtime_limit_seconds": float(
            get_nested(config, "runtime.stage_a_expected_seconds_upper_bound")
        ),
        "parent_prediction_logical_content_sha256": frozen["parent_logical_content_sha256"],
        "stage_a_prediction_logical_content_sha256": frozen["stage_a_logical_content_sha256"],
    }
    technical["passed"] = bool(
        technical["all_input_sha_matches"]
        and technical["prediction_rows"] == expected_rows
        and technical["prediction_wells"] == expected_wells
        and technical["reporting_folds"] == expected_folds
        and finite_coverage
        == float(technical_config["require_finite_control_and_candidate_coverage"])
        and bool(technical["same_x1p0_seed_bank"]["passed"])
        and abs(technical["temperature"] - 5.0)
        <= float(technical_config["temperature_absolute_tolerance"])
        and all(int(value) == 0 for value in before_freeze.values())
        and technical["all_stage_a_generation_and_training_counts_zero"]
        and bool(parity_metrics["passed"])
        and runtime_seconds <= technical["runtime_limit_seconds"]
    )

    direct_name = "direct_scale5_vs_arithmetic"
    blend_name = "fixed_exp209_hmm_likpf_50_50"
    overall = _metric_row(paired_metrics, direct_name, "overall")
    observed = _metric_row(paired_metrics, direct_name, "raw_gr_observed")
    fold_rows = paired_metrics.loc[
        paired_metrics["comparison"].eq(direct_name)
        & paired_metrics["scope"].str.startswith("fold_")
    ]
    improved_folds = int((fold_rows["delta_rmse_candidate_minus_control"] < 0.0).sum())
    regression_limits = {
        "raw_gr_missing": "maximum_raw_gr_missing_regression_ft",
        "missing_fraction_high": "maximum_high_missing_well_regression_ft",
        "md_since_1000_plus": "maximum_long_tail_1000_plus_regression_ft",
        "hidden_like_spatial": "maximum_hidden_like_spatial_regression_ft",
        "hidden_like_typewell_purged": ("maximum_hidden_like_typewell_purged_regression_ft"),
    }
    non_regression_scopes: dict[str, Any] = {}
    for scope, limit_key in regression_limits.items():
        delta = float(
            _metric_row(paired_metrics, direct_name, scope)["delta_rmse_candidate_minus_control"]
        )
        limit = float(scientific_config[limit_key])
        non_regression_scopes[scope] = {
            "delta_rmse_candidate_minus_control": delta,
            "maximum_regression_ft": limit,
            "passed": bool(delta <= limit),
        }
    direct_by_well = by_well_metrics.loc[by_well_metrics["comparison"].eq(direct_name)]
    by_well_delta = direct_by_well["delta_rmse_candidate_minus_control"]
    by_well_p95 = float(by_well_delta.quantile(0.95))
    worst_well_row = direct_by_well.loc[by_well_delta.idxmax()]
    blend_overall = _metric_row(paired_metrics, blend_name, "overall")
    blend_delta = float(blend_overall["delta_rmse_candidate_minus_control"])

    scientific_checks = {
        "pooled_gain": bool(
            float(overall["improvement_ft"])
            >= float(scientific_config["minimum_direct_rmse_gain_vs_arithmetic_mean_ft"])
        ),
        "improved_folds": bool(improved_folds >= int(scientific_config["minimum_improved_folds"])),
        "raw_gr_observed_gain": bool(
            float(observed["improvement_ft"])
            >= float(scientific_config["minimum_raw_gr_observed_gain_ft"])
        ),
        "all_required_scope_non_regression": bool(
            all(record["passed"] for record in non_regression_scopes.values())
        ),
        "by_well_delta_p95": bool(
            by_well_p95 <= float(scientific_config["maximum_by_well_delta_p95_ft"])
        ),
        "worst_well_regression": bool(
            float(worst_well_row["delta_rmse_candidate_minus_control"])
            <= float(scientific_config["maximum_worst_well_regression_ft"])
        ),
        "fixed_hmm_likpf_50_50_non_regression": bool(
            blend_delta
            <= float(scientific_config["maximum_fixed_hmm_likpf_50_50_blend_regression_ft"])
        ),
    }
    scientific = {
        "checks": scientific_checks,
        "candidate_rmse": float(overall["candidate_rmse"]),
        "control_rmse": float(overall["control_rmse"]),
        "gain_arithmetic_minus_scale5_ft": float(overall["improvement_ft"]),
        "minimum_gain_ft": float(
            scientific_config["minimum_direct_rmse_gain_vs_arithmetic_mean_ft"]
        ),
        "improved_folds": improved_folds,
        "minimum_improved_folds": int(scientific_config["minimum_improved_folds"]),
        "raw_gr_observed_gain_ft": float(observed["improvement_ft"]),
        "minimum_raw_gr_observed_gain_ft": float(
            scientific_config["minimum_raw_gr_observed_gain_ft"]
        ),
        "non_regression_scopes": non_regression_scopes,
        "by_well_delta_p95_ft": by_well_p95,
        "maximum_by_well_delta_p95_ft": float(scientific_config["maximum_by_well_delta_p95_ft"]),
        "worst_well_id": str(worst_well_row["well_id"]),
        "worst_well_regression_ft": float(worst_well_row["delta_rmse_candidate_minus_control"]),
        "maximum_worst_well_regression_ft": float(
            scientific_config["maximum_worst_well_regression_ft"]
        ),
        "fixed_hmm_likpf_50_50_delta_rmse_ft": blend_delta,
        "maximum_fixed_hmm_likpf_50_50_regression_ft": float(
            scientific_config["maximum_fixed_hmm_likpf_50_50_blend_regression_ft"]
        ),
        "passed": bool(all(scientific_checks.values())),
    }
    passed = bool(technical["passed"] and scientific["passed"])
    decision = get_nested(config, "guards.decision") or {}
    return {
        "experiment": EXPERIMENT_NAME,
        "passed": passed,
        "decision": str(decision["pass_action"] if passed else decision["fail_action"]),
        "technical_gate": technical,
        "scientific_gate": scientific,
        "primary_policy": "fixed_temperature_5_x1p0_vs_arithmetic_x1p0_same_seed_bank",
        "batch_semantics": get_nested(
            config,
            "validation.batch_inference_semantics",
        ),
        "failure_action": (
            "close_without_temperature_scale_best_seed_median_mode_medoid_selector_"
            "roughening_sigma_model_blend_gate_or_same_oof_rescue"
        ),
    }


# %% [markdown]
# ## 8. Generated artifacts and execution orchestration


# %%
def input_manifest_frame(preflight: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, report in preflight["reports"].items():
        rows.append(
            {
                "name": name,
                "path": report.get("path"),
                "bytes": report.get("bytes"),
                "raw_sha256": report.get("raw_sha256"),
                "decompressed_sha256": report.get("decompressed_sha256"),
                "scientific_contract_sha256": report.get("scientific_contract_sha256"),
                "data_rows": report.get("data_rows"),
                "columns": json.dumps(report.get("columns"), separators=(",", ":")),
            }
        )
    return pd.DataFrame(rows).sort_values("name", kind="mergesort").reset_index(drop=True)


def artifact_report(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": sha256_path(path),
    }


def build_artifact_manifest(paths: dict[str, Path]) -> pd.DataFrame:
    rows = [{"name": name, **artifact_report(path)} for name, path in paths.items()]
    return pd.DataFrame(rows).sort_values("name", kind="mergesort").reset_index(drop=True)


def run_full_experiment(config: dict[str, Any]) -> dict[str, Any]:
    if not KAGGLE_WORKING_ROOT.exists() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError(
            "Full exp417 must run first on Kaggle; local execution requires explicit smoke approval"
        )
    scientific_contract = validate_scientific_contract(
        config,
        require_run_approval=True,
    )
    started = time.time()
    artifacts = artifact_dir()
    preflight = preflight_saved_inputs(config)
    ledger = TruthAccessLedger()
    contract_path = artifacts / f"{OUTPUT_PREFIX}_scientific_contract.json"
    input_manifest_path = artifacts / f"{OUTPUT_PREFIX}_input_manifest.csv"
    write_json(contract_path, scientific_contract)
    input_manifest_frame(preflight).to_csv(input_manifest_path, index=False)

    prediction, _audit, frozen = load_and_freeze_prediction_identity(
        preflight,
        config,
        ledger,
    )
    prediction_frozen_at_seconds = time.time() - started
    frozen_identity_path = artifacts / f"{OUTPUT_PREFIX}_frozen_prediction_identity.json"
    write_json(frozen_identity_path, frozen)
    frame, late_attachment = load_late_readout_frame(
        prediction,
        frozen,
        preflight,
        config,
        ledger,
    )
    paired_metrics, by_well_metrics = build_metric_outputs(frame)
    parity_metrics = build_parity_metrics(frame, config)
    runtime_seconds = time.time() - started
    promotion_gate = evaluate_promotion_gate(
        frame,
        paired_metrics,
        by_well_metrics,
        parity_metrics,
        preflight,
        frozen,
        ledger,
        runtime_seconds,
        config,
    )

    metric_paths = {
        "paired_metrics": artifacts / f"{OUTPUT_PREFIX}_paired_metrics.csv",
        "by_well_metrics": artifacts / f"{OUTPUT_PREFIX}_by_well_metrics.csv",
        "parity_metrics": artifacts / f"{OUTPUT_PREFIX}_parity_metrics.json",
        "promotion_gate": artifacts / f"{OUTPUT_PREFIX}_promotion_gate.json",
    }
    paired_metrics.to_csv(metric_paths["paired_metrics"], index=False)
    by_well_metrics.to_csv(metric_paths["by_well_metrics"], index=False)
    write_json(metric_paths["parity_metrics"], parity_metrics)
    write_json(metric_paths["promotion_gate"], promotion_gate)
    manifest_sources = {
        **metric_paths,
        "scientific_contract": contract_path,
        "frozen_prediction_identity": frozen_identity_path,
        "input_manifest": input_manifest_path,
    }
    artifact_manifest = build_artifact_manifest(manifest_sources)
    artifact_manifest_path = artifacts / f"{OUTPUT_PREFIX}_artifact_manifest.csv"
    artifact_manifest.to_csv(artifact_manifest_path, index=False)
    artifact_manifest_sha = sha256_path(artifact_manifest_path)
    status = (
        "stage_a_passed_raw_test_inference_design_approval_pending"
        if promotion_gate["passed"]
        else "stage_a_failed_closed_no_rescue"
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "route": "pf_beam",
        "runtime_seconds": runtime_seconds,
        "prediction_frozen_at_seconds": prediction_frozen_at_seconds,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "active_scientific_candidates": 1,
        "saved_candidate_readouts": 1,
        "pf_well_runs": 0,
        "parent_pf_control_reruns": 0,
        "models": 0,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
        "scientific_contract_sha256": scientific_contract["scientific_contract_sha256"],
        "input_manifest_sha256": sha256_path(input_manifest_path),
        "artifact_manifest_sha256": artifact_manifest_sha,
        "frozen_prediction": frozen,
        "truth_attachment": late_attachment,
        "parity_metrics": parity_metrics,
        "promotion_gate": promotion_gate,
        "runtime_versions": runtime_versions(),
        "kaggle": {
            "kernel_version": None,
            "kernel_version_recording": "record_from_kaggle_api_after_run",
            "kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        },
        "model_sha256": None,
        "submission_sha256": None,
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"
    write_json(summary_path, summary)
    overall = _metric_row(
        paired_metrics,
        "direct_scale5_vs_arithmetic",
        "overall",
    )
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "route": "pf_beam",
        "cv": float(overall["candidate_rmse"]),
        "public_lb": None,
        "private_lb": None,
        "metric": "rmse",
        "overall": overall.to_dict(),
        "parity_metrics": parity_metrics,
        "promotion_gate": promotion_gate,
        "prediction_sha256": frozen["stage_a_logical_content_sha256"],
        "parent_prediction_sha256": frozen["parent_logical_content_sha256"],
        "artifact_manifest_sha256": artifact_manifest_sha,
        "model_sha256": None,
        "submission_sha256": None,
        "notes": (
            "Saved-OOF fixed scale-5 versus arithmetic x1p0 seed aggregation audit. "
            "No PF/HMM/model rerun, raw-test prediction, inference, or submission."
        ),
    }
    write_json(metrics_output_path(), metrics)
    print(paired_metrics.to_string(index=False))
    print(json.dumps(to_jsonable(parity_metrics), indent=2, sort_keys=True))
    print(json.dumps(to_jsonable(promotion_gate), indent=2, sort_keys=True))
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 9. Setup and configuration preview


# %%
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    SCIENTIFIC_CONTRACT = validate_scientific_contract(CONFIG)
    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "route": get_nested(CONFIG, "experiment.route"),
                "artifact_parent": get_nested(CONFIG, "lineage.parent"),
                "scientific_control": get_nested(
                    CONFIG,
                    "lineage.scientific_control",
                ),
                "primary_control": PRIMARY_CONTROL,
                "primary_candidate": PRIMARY_CANDIDATE,
                "temperature": get_nested(
                    CONFIG,
                    "model.aggregation.candidate.temperature",
                ),
                "execution_counts": get_nested(
                    CONFIG,
                    "model.execution_count",
                ),
                "batch_non_causal": True,
                "canonical_notebook_adoption_approved": get_nested(
                    CONFIG,
                    "execution.canonical_notebook_adoption_approved",
                ),
                "kaggle_package_approved": get_nested(
                    CONFIG,
                    "execution.kaggle_package_approved",
                ),
                "stage_a_run_approved": get_nested(
                    CONFIG,
                    "execution.stage_a_run_approved",
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


# %% [markdown]
# ## 10. Run the Kaggle CPU audit


# %%
if EXECUTE_NOTEBOOK:
    SUMMARY = run_full_experiment(CONFIG)
