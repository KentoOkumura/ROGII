# %% [markdown]
# # exp422 roughening ×10 failure-regime attribution readout — train
#
# This is a saved-output, zero-PF attribution audit of the terminal exp416
# roughening-x10 failure. Before any truth, control outcome, by-well outcome, or
# persistent-episode outcome is opened, it freezes one preregistered two-axis
# well regime and three fixed row scopes. A PASS supports only the design of a
# separate policy experiment; it never reclassifies exp416.

# %% [markdown]
# ## Contents
# 1. Imports and fixed notebook contract
# 2. Notebook-safe configuration, path, and SHA helpers
# 3. Frozen source, execution, and attribution contract
# 4. Source preflight and target-free input checks
# 5. Fold-safe regime and row-scope freeze
# 6. Late outcome attachment and parent-metric parity
# 7. Association, regime, position, and episode readouts
# 8. Technical and scientific attribution gates
# 9. Generated artifacts and execution orchestration
# 10. Setup and configuration preview
# 11. Run the approved Kaggle CPU readout

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

EXPERIMENT_NAME = "exp422_roughening_x10_failure_regime_attribution_readout"
OUTPUT_PREFIX = EXPERIMENT_NAME
PRIMARY_CANDIDATE = "likpf_roughening_x10_mean"
PRIMARY_CONTROL = "saved_exp072_likpf_mean"
TRUE_TVT = "tvt_true"
TARGET_CELL = "high_recovery_pressure__low_damage_exposure"
RECOVERY_COMPONENTS = (
    "resampling_rate",
    "ess_collapse",
    "seed_prediction_dispersion",
    "seed_likelihood_gap",
)
DAMAGE_COMPONENTS = (
    "eval_missing_fraction",
    "suffix_horizon",
)
RAW_DIAGNOSTICS = (*RECOVERY_COMPONENTS, *DAMAGE_COMPONENTS)
SECONDARY_MEDIATORS = (
    "position_clip_rate",
    "prefix_missing_fraction",
    "gr_scale_clipped",
)
PARENT_PREDICTION_COLUMNS = (
    "id",
    "well_id",
    "row_idx",
    "suffix_offset",
    "last_known_tvt",
    "md_since",
    "raw_gr_observed",
    PRIMARY_CANDIDATE,
)
PARENT_LOGICAL_COLUMNS = (
    "id",
    "well_id",
    "row_idx",
    PRIMARY_CANDIDATE,
)
REGIME_FEATURE_COLUMNS = (
    "well_id",
    "fold",
    *RAW_DIAGNOSTICS,
    *SECONDARY_MEDIATORS,
    *(f"{name}_ecdf" for name in RAW_DIAGNOSTICS),
    "recovery_pressure_score",
    "damage_exposure_score",
    "recovery_pressure_outer_median",
    "damage_exposure_outer_median",
    "high_recovery_pressure",
    "low_damage_exposure",
    "regime_cell",
    "is_primary_target_cell",
)
ROW_SCOPE_COLUMNS = (
    "id",
    "well_id",
    "row_idx",
    "suffix_offset",
    "fold",
    "normalized_suffix_progress",
    "suffix_progress_quartile",
    "raw_gr_status",
    "md_since_1000_ft",
    "regime_cell",
    "is_primary_target_cell",
)
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
SOURCE_FILENAME = f"{EXPERIMENT_NAME}_compact_selfcontained_train.py"


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = (
    os.environ.get("EXP422_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()
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


def canonical_json(value: Any) -> str:
    return json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def mapping_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


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
    raise FileNotFoundError(f"exp422 config not found; checked={checked}")


def resolve_package_file(filename: str) -> Path:
    checked: list[str] = []
    for package_dir in candidate_package_dirs():
        path = package_dir / filename
        checked.append(str(path))
        if path.exists() and path.is_file():
            return path
    if KAGGLE_INPUT_ROOT.exists():
        matches = sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}"))
        if len(matches) == 1:
            return matches[0]
    raise FileNotFoundError(f"exp422 package file {filename} not found; checked={checked}")


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
        matches = sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}"))
        if len(matches) == 1:
            return matches[0]
    raise FileNotFoundError(f"could not resolve {filename}; checked={checked}")


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
        "data_rows": max(0, line_count - 1),
        "columns": pd.read_csv(csv_path, nrows=0).columns.astype(str).tolist(),
    }


def write_deterministic_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )


def dataframe_content_sha(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
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


def stable_seed(*parts: object, modulo: int = 2_147_483_647) -> int:
    key = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo + 1


def parse_bool_series(values: pd.Series, *, label: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    if not normalized.isin(mapping).all():
        invalid = sorted(normalized.loc[~normalized.isin(mapping)].unique().tolist())
        raise ValueError(f"{label} contains invalid booleans: {invalid}")
    return normalized.map(mapping).astype(bool)


def runtime_versions() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyyaml": getattr(yaml, "__version__", "unknown"),
    }


def maximum_rss_gb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 * 1024.0)


# %% [markdown]
# ## 3. Frozen source, execution, and attribution contract


# %%
def execution_counts(config: Mapping[str, Any]) -> dict[str, int]:
    keys = (
        "saved_output_readout_contracts",
        "new_prediction_rows",
        "scientific_pf_variants",
        "candidate_pf_well_runs",
        "parent_pf_control_reruns",
        "lightgbm_configs",
        "trained_folds",
        "boosters",
        "hmm_well_runs",
        "beam_well_runs",
        "gpu_runs",
        "reporting_folds",
    )
    return {key: int(get_nested(config, f"execution.{key}")) for key in keys}


def build_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "parent": get_nested(config, "lineage.parent"),
        "parent_terminal_decision": get_nested(
            config, "data.exp416_terminal_contract.decision"
        ),
        "source_kernel": {
            "slug": get_nested(config, "data.exp416_merge_output.kernel_source"),
            "id_no": get_nested(config, "data.exp416_merge_output.source_kernel_id_no"),
            "version": get_nested(config, "data.exp416_merge_output.source_kernel_version"),
            "artifact_manifest_raw_sha256": get_nested(
                config,
                "data.exp416_merge_output.expected_artifact_manifest_raw_sha256",
            ),
        },
        "target_free_freeze": {
            "outer_reference_policy": get_nested(
                config, "readout.outer_reference_policy"
            ),
            "empirical_cdf_formula": get_nested(
                config, "readout.empirical_cdf_formula"
            ),
            "recovery_components": list(RECOVERY_COMPONENTS),
            "damage_components": list(DAMAGE_COMPONENTS),
            "score_aggregation": "equal_weight_mean",
            "recovery_high_rule": get_nested(
                config, "readout.scores.recovery_pressure.high_rule"
            ),
            "damage_low_rule": get_nested(
                config, "readout.scores.damage_exposure.low_rule"
            ),
            "primary_target_cell": TARGET_CELL,
        },
        "permutation": {
            "count": get_nested(config, "guards.scientific.permutation_count"),
            "scope": get_nested(config, "guards.scientific.permutation_scope"),
            "seed_policy": get_nested(config, "reproducibility.seed_policy"),
        },
        "execution_counts": execution_counts(config),
        "forbidden": list(get_nested(config, "guards.forbidden") or []),
        "parent_decision_remains_unchanged": True,
        "policy_ready_if_passed": False,
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
        "lineage.parent": "exp416_roughening_x10_likpf_full_oof_ablation",
        "implementation.enabled": True,
        "implementation.scope": "saved_output_failure_regime_attribution_readout",
        "model.estimator": "none",
        "model.fitted_model": False,
        "validation.expected_rows": 3_783_989,
        "validation.expected_wells": 773,
        "validation.expected_folds": [0, 1, 2, 3, 4],
        "validation.expected_persistent_episode_wells": 12,
        "validation.expected_persistent_episodes": 16,
        "validation.expected_persistent_episode_rows": 55_104,
        "data.exp416_merge_output.source_kernel_id_no": 128_912_230,
        "data.exp416_merge_output.source_kernel_version": 2,
        "data.exp416_terminal_contract.status": (
            "train_side_roughening_x10_full_oof_gate_failed_closed"
        ),
        "data.exp416_terminal_contract.decision": (
            "roughening_x10_rejected_close_without_rescue"
        ),
        "data.exp416_terminal_contract.scientific_gate_passed": False,
        "data.exp416_terminal_contract.technical_gate_passed": False,
        "readout.primary_target_cell": TARGET_CELL,
        "readout.scores.recovery_pressure.components": list(RECOVERY_COMPONENTS),
        "readout.scores.damage_exposure.components": list(DAMAGE_COMPONENTS),
        "guards.scientific.permutation_count": 4096,
        "guards.scientific.permutation_scope": "within_reporting_fold",
        "execution.saved_output_readout_contracts": 1,
        "execution.new_prediction_rows": 0,
        "execution.scientific_pf_variants": 0,
        "execution.candidate_pf_well_runs": 0,
        "execution.parent_pf_control_reruns": 0,
        "execution.lightgbm_configs": 0,
        "execution.trained_folds": 0,
        "execution.boosters": 0,
        "execution.hmm_well_runs": 0,
        "execution.beam_well_runs": 0,
        "execution.gpu_runs": 0,
        "execution.reporting_folds": 5,
        "runtime.device": "cpu",
        "runtime.kaggle.enable_gpu": False,
        "runtime.kaggle.enable_internet": False,
        "inference.enabled": False,
        "execution.inference_approved": False,
        "execution.submission_approved": False,
    }
    for key, value in expected.items():
        if get_nested(config, key) != value:
            raise ValueError(f"exp422 fixed contract mismatch: {key} must be {value!r}")
    if not bool(get_nested(config, "execution.implementation_approved")):
        raise ValueError("exp422 implementation approval must be recorded")
    if require_run_approval and not (
        bool(get_nested(config, "execution.canonical_notebook_adoption_approved"))
        and bool(get_nested(config, "execution.kaggle_package_approved"))
        and bool(get_nested(config, "execution.kaggle_push_approved"))
        and bool(get_nested(config, "execution.audit_run_approved"))
    ):
        raise RuntimeError("exp422 canonical notebook/package/push/audit run is not approved")
    return build_scientific_contract(config)


def validate_parent_terminal_contract(
    parent_contract: Mapping[str, Any],
    parent_gate: Mapping[str, Any],
    parent_summary: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    expected_contract_sha = str(
        get_nested(
            config,
            "data.exp416_merge_output.expected_scientific_contract_sha256",
        )
    )
    expected_status = str(get_nested(config, "data.exp416_terminal_contract.status"))
    expected_decision = str(get_nested(config, "data.exp416_terminal_contract.decision"))
    manifest_sha = str(
        get_nested(
            config,
            "data.exp416_merge_output.expected_artifact_manifest_raw_sha256",
        )
    )
    checks = {
        "scientific_contract_sha": (
            str(parent_contract.get("scientific_contract_sha256"))
            == expected_contract_sha
        ),
        "gate_failed": parent_gate.get("passed") is False,
        "gate_decision": str(parent_gate.get("decision")) == expected_decision,
        "summary_status": str(parent_summary.get("status")) == expected_status,
        "summary_gate_failed": get_nested(parent_summary, "gate.passed") is False,
        "summary_gate_decision": (
            str(get_nested(parent_summary, "gate.decision")) == expected_decision
        ),
        "summary_manifest_sha": (
            str(parent_summary.get("artifact_manifest_sha256")) == manifest_sha
        ),
    }
    if not all(checks.values()):
        raise ValueError(f"exp416 terminal contract mismatch: {checks}")
    return {
        "passed": True,
        "checks": checks,
        "status": expected_status,
        "decision": expected_decision,
        "parent_reclassified": False,
    }


# %% [markdown]
# ## 4. Source preflight and target-free input checks


# %%
def _source_spec(config: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = get_nested(config, f"data.{key}") or {}
    if not isinstance(value, dict):
        raise ValueError(f"data.{key} must be a mapping")
    return dict(value)


def _manifest_row_for_filename(manifest: pd.DataFrame, filename: str) -> pd.Series:
    if "path" not in manifest or "raw_sha256" not in manifest:
        raise ValueError("exp416 artifact manifest lacks path/raw_sha256")
    matches = manifest.loc[
        manifest["path"].astype(str).map(lambda value: Path(value).name == filename)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one manifest row for {filename}; found={len(matches)}")
    return matches.iloc[0]


def _validate_manifest_file(
    manifest: pd.DataFrame,
    path: Path,
) -> dict[str, Any]:
    row = _manifest_row_for_filename(manifest, path.name)
    observed = sha256_path(path)
    expected = str(row["raw_sha256"])
    if observed != expected:
        raise ValueError(f"exp416 manifest SHA mismatch for {path.name}")
    report: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": observed,
    }
    if path.suffix == ".gz":
        report.update(inspect_gzip_csv(path))
        manifest_decompressed = row.get("decompressed_sha256")
        if pd.notna(manifest_decompressed) and str(manifest_decompressed):
            if report["decompressed_sha256"] != str(manifest_decompressed):
                raise ValueError(
                    f"exp416 manifest decompressed SHA mismatch for {path.name}"
                )
    elif path.suffix == ".csv":
        report["columns"] = pd.read_csv(path, nrows=0).columns.astype(str).tolist()
    return report


def preflight_saved_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    parent = _source_spec(config, "exp416_merge_output")
    exp072 = _source_spec(config, "exp072_control")
    exp226 = _source_spec(config, "exp226_reporting")
    parent_candidates = [str(value) for value in parent.get("candidates", [])]
    manifest_path = resolve_existing(
        str(parent["artifact_manifest_filename"]),
        parent_candidates,
    )
    manifest_sha = sha256_path(manifest_path)
    if manifest_sha != str(parent["expected_artifact_manifest_raw_sha256"]):
        raise ValueError("exp416 artifact manifest raw SHA mismatch")
    manifest = pd.read_csv(manifest_path)

    parent_files = {
        "parent_contract": str(parent["scientific_contract_filename"]),
        "parent_gate": str(parent["scientific_gate_filename"]),
        "parent_prediction": str(parent["merged_prediction_filename"]),
        "parent_well_audit": str(parent["merged_well_audit_filename"]),
        "parent_by_well": str(parent["by_well_metrics_filename"]),
        "parent_episodes": str(parent["persistent_episode_metrics_filename"]),
    }
    paths = {
        name: resolve_existing(filename, parent_candidates)
        for name, filename in parent_files.items()
    }
    paths["parent_summary"] = resolve_existing(
        str(parent["summary_filename"]),
        parent_candidates,
    )
    reports = {
        name: _validate_manifest_file(manifest, path)
        for name, path in paths.items()
        if name != "parent_summary"
    }
    prediction_report = reports["parent_prediction"]
    fixed_prediction_checks = {
        "raw_sha256": str(parent["expected_prediction_raw_gzip_sha256"]),
        "decompressed_sha256": str(parent["expected_prediction_decompressed_sha256"]),
    }
    for key, expected in fixed_prediction_checks.items():
        if str(prediction_report[key]) != expected:
            raise ValueError(f"exp416 prediction {key} mismatch")

    parent_contract = json.loads(paths["parent_contract"].read_text())
    parent_gate = json.loads(paths["parent_gate"].read_text())
    parent_summary = json.loads(paths["parent_summary"].read_text())
    terminal_contract = validate_parent_terminal_contract(
        parent_contract,
        parent_gate,
        parent_summary,
        config,
    )

    paths["exp072_control"] = resolve_existing(
        str(exp072["filename"]),
        [str(value) for value in exp072.get("candidates", [])],
    )
    reports["exp072_control"] = inspect_gzip_csv(paths["exp072_control"])
    if (
        reports["exp072_control"]["raw_sha256"]
        != str(exp072["expected_raw_gzip_sha256"])
        or reports["exp072_control"]["decompressed_sha256"]
        != str(exp072["expected_decompressed_sha256"])
    ):
        raise ValueError("exp072 control SHA mismatch")

    paths["exp226_reporting"] = resolve_existing(
        str(exp226["filename"]),
        [str(value) for value in exp226.get("candidates", [])],
    )
    reports["exp226_reporting"] = inspect_gzip_csv(paths["exp226_reporting"])
    if reports["exp226_reporting"]["decompressed_sha256"] != str(
        exp226["expected_decompressed_sha256"]
    ):
        raise ValueError("exp226 reporting decompressed SHA mismatch")

    required_columns = {
        "parent_prediction": set(PARENT_PREDICTION_COLUMNS),
        "parent_well_audit": {
            "well_id",
            "status",
            "prefix_rows",
            "prefix_gr_missing_rows",
            "eval_rows",
            "eval_raw_gr_missing_rows",
            "seeds",
            "particles",
            "seed_loglik_mean_per_row",
            "seed_loglik_best_per_row",
            "resampling_count_total",
            "minimum_ess_mean",
            "position_clip_count_total",
            "seed_prediction_std_mean",
            "gr_scale_clipped",
        },
        "parent_by_well": {
            "well_id",
            "rows",
            "candidate_rmse",
            "control_rmse",
            "improvement_ft",
        },
        "parent_episodes": {
            "episode_id",
            "well_id",
            "rows",
            "candidate_sse",
            "control_sse",
            "sse_reduction_fraction",
            "improved",
        },
        "exp072_control": {
            "id",
            *[str(value) for value in exp072["reconstruction_columns"]],
        },
        "exp226_reporting": {
            *[str(value) for value in exp226["prefreeze_safe_columns"]],
            *[str(value) for value in exp226["postfreeze_truth_columns"]],
        },
    }
    for name, required in required_columns.items():
        missing = sorted(required - set(reports[name]["columns"]))
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}")
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    for name in ("parent_prediction", "exp072_control", "exp226_reporting"):
        if int(reports[name]["data_rows"]) != expected_rows:
            raise ValueError(f"{name} row count mismatch")
    return {
        "paths": {name: str(path) for name, path in paths.items()},
        "reports": reports,
        "parent_manifest": {
            "path": str(manifest_path),
            "rows": len(manifest),
            "raw_sha256": manifest_sha,
        },
        "parent_terminal_contract": terminal_contract,
        "source_kernel": {
            "slug": str(parent["kernel_source"]),
            "id_no": int(parent["source_kernel_id_no"]),
            "version": int(parent["source_kernel_version"]),
            "status": str(parent["source_kernel_status"]),
        },
        "all_input_sha_matches": True,
        "outcome_rows_parsed_before_regime_freeze": {
            "truth": 0,
            "control": 0,
            "by_well": 0,
            "persistent_episode": 0,
        },
    }


@dataclass
class OutcomeAccessLedger:
    regime_frozen: bool = False
    prefreeze_well_audit_rows: int = 0
    prefreeze_prediction_identity_rows: int = 0
    prefreeze_reporting_identity_rows: int = 0
    truth_rows_before_freeze: int = 0
    control_rows_before_freeze: int = 0
    by_well_rows_before_freeze: int = 0
    persistent_episode_rows_before_freeze: int = 0
    truth_rows_after_freeze: int = 0
    control_rows_after_freeze: int = 0
    by_well_rows_after_freeze: int = 0
    persistent_episode_rows_after_freeze: int = 0

    def mark_frozen(self) -> None:
        if self.regime_frozen:
            raise RuntimeError("target-free regime is already frozen")
        if any(self.report()["outcome_before_freeze"].values()):
            raise RuntimeError("outcome rows were opened before target-free regime freeze")
        self.regime_frozen = True

    def require_frozen(self) -> None:
        if not self.regime_frozen:
            raise RuntimeError("late outcome attachment requires a frozen target-free regime")

    def report(self) -> dict[str, Any]:
        return {
            "regime_frozen": self.regime_frozen,
            "prefreeze_safe_rows": {
                "well_audit": self.prefreeze_well_audit_rows,
                "prediction_identity": self.prefreeze_prediction_identity_rows,
                "reporting_identity": self.prefreeze_reporting_identity_rows,
            },
            "outcome_before_freeze": {
                "truth": self.truth_rows_before_freeze,
                "control": self.control_rows_before_freeze,
                "by_well": self.by_well_rows_before_freeze,
                "persistent_episode": self.persistent_episode_rows_before_freeze,
            },
            "outcome_after_freeze": {
                "truth": self.truth_rows_after_freeze,
                "control": self.control_rows_after_freeze,
                "by_well": self.by_well_rows_after_freeze,
                "persistent_episode": self.persistent_episode_rows_after_freeze,
            },
        }


def load_target_free_inputs(
    preflight: Mapping[str, Any],
    config: Mapping[str, Any],
    ledger: OutcomeAccessLedger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if ledger.regime_frozen:
        raise RuntimeError("target-free inputs cannot be reopened after freeze")
    prediction = pd.read_csv(
        preflight["paths"]["parent_prediction"],
        usecols=list(PARENT_PREDICTION_COLUMNS),
        dtype={
            "id": str,
            "well_id": str,
            "row_idx": np.int64,
            "suffix_offset": np.int64,
            PRIMARY_CANDIDATE: np.float32,
        },
    )
    prediction = prediction[list(PARENT_PREDICTION_COLUMNS)]
    prediction["id"] = prediction["id"].astype(object)
    prediction["well_id"] = prediction["well_id"].astype(object)
    prediction["last_known_tvt"] = pd.to_numeric(
        prediction["last_known_tvt"], errors="raise"
    ).astype(np.float64)
    prediction["md_since"] = pd.to_numeric(
        prediction["md_since"], errors="raise"
    ).astype(np.float64)
    prediction["raw_gr_observed"] = parse_bool_series(
        prediction["raw_gr_observed"],
        label="exp416 raw_gr_observed",
    )
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    logical_columns = list(PARENT_LOGICAL_COLUMNS)
    if (
        len(prediction) != expected_rows
        or prediction["well_id"].nunique() != expected_wells
        or prediction["id"].duplicated().any()
        or prediction.duplicated(["well_id", "row_idx"]).any()
    ):
        raise ValueError("exp416 prediction identity coverage mismatch")
    if dataframe_content_sha(prediction, logical_columns) != str(
        get_nested(
            config,
            "data.exp416_merge_output.expected_prediction_logical_content_sha256",
        )
    ):
        raise ValueError("exp416 prediction logical content SHA mismatch")
    if dataframe_schema_sha(prediction) != str(
        get_nested(config, "data.exp416_merge_output.expected_prediction_schema_sha256")
    ):
        raise ValueError("exp416 prediction schema SHA mismatch")
    if not np.isfinite(
        prediction[["last_known_tvt", "md_since", PRIMARY_CANDIDATE]].to_numpy(
            np.float64
        )
    ).all():
        raise ValueError("exp416 prediction contains non-finite safe values")
    ledger.prefreeze_prediction_identity_rows += len(prediction)

    audit = pd.read_csv(
        preflight["paths"]["parent_well_audit"],
        dtype={"well_id": str},
    ).sort_values("well_id", kind="mergesort").reset_index(drop=True)
    if (
        len(audit) != expected_wells
        or audit["well_id"].duplicated().any()
        or not audit["status"].eq("ok").all()
    ):
        raise ValueError("exp416 well-audit coverage mismatch")
    ledger.prefreeze_well_audit_rows += len(audit)

    safe_columns = [
        str(value)
        for value in get_nested(config, "data.exp226_reporting.prefreeze_safe_columns")
    ]
    reporting = pd.read_csv(
        preflight["paths"]["exp226_reporting"],
        usecols=safe_columns,
        dtype={"well_id": str},
    )
    for column in ("row_idx", "suffix_offset", "fold"):
        reporting[column] = pd.to_numeric(
            reporting[column], errors="raise"
        ).astype(np.int64)
    reporting = reporting.sort_values(
        ["well_id", "row_idx"], kind="mergesort"
    ).reset_index(drop=True)
    prediction = prediction.sort_values(
        ["well_id", "row_idx"], kind="mergesort"
    ).reset_index(drop=True)
    if (
        len(reporting) != expected_rows
        or reporting.duplicated(["well_id", "row_idx"]).any()
        or not np.array_equal(
            reporting[["well_id", "row_idx"]].to_numpy(),
            prediction[["well_id", "row_idx"]].to_numpy(),
        )
        or not np.array_equal(
            reporting["suffix_offset"].to_numpy(np.int64),
            prediction["suffix_offset"].to_numpy(np.int64),
        )
    ):
        raise ValueError("exp226 prefreeze reporting identity mismatch")
    expected_folds = [
        int(value) for value in get_nested(config, "validation.expected_folds")
    ]
    if sorted(reporting["fold"].unique().tolist()) != expected_folds:
        raise ValueError("reporting fold set mismatch")
    fold_counts = reporting.groupby("well_id", sort=False)["fold"].nunique()
    if not fold_counts.eq(1).all():
        raise ValueError("reporting fold must be constant within each well")
    ledger.prefreeze_reporting_identity_rows += len(reporting)
    return prediction, audit, reporting


# %% [markdown]
# ## 5. Fold-safe regime and row-scope freeze


# %%
def build_raw_diagnostics(audit: pd.DataFrame) -> pd.DataFrame:
    frame = audit.copy()
    numeric_columns = (
        "prefix_rows",
        "prefix_gr_missing_rows",
        "eval_rows",
        "eval_raw_gr_missing_rows",
        "seeds",
        "particles",
        "seed_loglik_mean_per_row",
        "seed_loglik_best_per_row",
        "resampling_count_total",
        "minimum_ess_mean",
        "position_clip_count_total",
        "seed_prediction_std_mean",
        "gr_scale_clipped",
    )
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if (
        frame["eval_rows"].le(0).any()
        or frame["prefix_rows"].le(0).any()
        or frame["seeds"].le(0).any()
        or frame["particles"].le(0).any()
    ):
        raise ValueError("diagnostic denominator must be positive")
    seed_eval = frame["seeds"] * frame["eval_rows"]
    frame["resampling_rate"] = frame["resampling_count_total"] / seed_eval
    frame["ess_collapse"] = 1.0 - np.clip(
        frame["minimum_ess_mean"] / frame["particles"],
        0.0,
        1.0,
    )
    frame["seed_prediction_dispersion"] = np.log1p(
        frame["seed_prediction_std_mean"]
    )
    frame["seed_likelihood_gap"] = np.log1p(
        np.maximum(
            frame["seed_loglik_best_per_row"] - frame["seed_loglik_mean_per_row"],
            0.0,
        )
    )
    frame["eval_missing_fraction"] = (
        frame["eval_raw_gr_missing_rows"] / frame["eval_rows"]
    )
    frame["suffix_horizon"] = np.log1p(frame["eval_rows"])
    frame["position_clip_rate"] = frame["position_clip_count_total"] / seed_eval
    frame["prefix_missing_fraction"] = (
        frame["prefix_gr_missing_rows"] / frame["prefix_rows"]
    )
    diagnostics = frame[
        ["well_id", "eval_rows", *RAW_DIAGNOSTICS, *SECONDARY_MEDIATORS]
    ].copy()
    if not np.isfinite(
        diagnostics[[*RAW_DIAGNOSTICS, *SECONDARY_MEDIATORS]].to_numpy(np.float64)
    ).all():
        raise ValueError("target-free diagnostic contains non-finite values")
    return diagnostics


def empirical_cdf(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference_values = np.asarray(reference, dtype=np.float64)
    target_values = np.asarray(values, dtype=np.float64)
    if reference_values.size == 0 or not np.isfinite(reference_values).all():
        raise ValueError("empirical CDF reference must be finite and nonempty")
    if not np.isfinite(target_values).all():
        raise ValueError("empirical CDF target values must be finite")
    ordered = np.sort(reference_values, kind="mergesort")
    return np.searchsorted(ordered, target_values, side="right") / len(ordered)


def build_fold_safe_regime_features(
    audit: pd.DataFrame,
    reporting: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    diagnostics = build_raw_diagnostics(audit)
    well_fold = (
        reporting[["well_id", "fold"]]
        .drop_duplicates()
        .sort_values("well_id", kind="mergesort")
    )
    if well_fold["well_id"].duplicated().any():
        raise ValueError("well-fold identity is duplicated")
    features = diagnostics.merge(
        well_fold,
        on="well_id",
        how="left",
        validate="one_to_one",
    )
    if features["fold"].isna().any():
        raise ValueError("well audit is missing reporting fold")
    features["fold"] = features["fold"].astype(np.int64)
    expected_folds = [
        int(value) for value in get_nested(config, "validation.expected_folds")
    ]
    parts: list[pd.DataFrame] = []
    for fold in expected_folds:
        target = features.loc[features["fold"].eq(fold)].copy()
        reference = features.loc[features["fold"].ne(fold)].copy()
        if target.empty or reference.empty:
            raise ValueError(f"fold {fold} lacks target or outer reference wells")
        for diagnostic in RAW_DIAGNOSTICS:
            target[f"{diagnostic}_ecdf"] = empirical_cdf(
                reference[diagnostic].to_numpy(np.float64),
                target[diagnostic].to_numpy(np.float64),
            )
            reference[f"{diagnostic}_ecdf"] = empirical_cdf(
                reference[diagnostic].to_numpy(np.float64),
                reference[diagnostic].to_numpy(np.float64),
            )
        target["recovery_pressure_score"] = target[
            [f"{name}_ecdf" for name in RECOVERY_COMPONENTS]
        ].mean(axis=1)
        target["damage_exposure_score"] = target[
            [f"{name}_ecdf" for name in DAMAGE_COMPONENTS]
        ].mean(axis=1)
        reference_recovery = reference[
            [f"{name}_ecdf" for name in RECOVERY_COMPONENTS]
        ].mean(axis=1)
        reference_damage = reference[
            [f"{name}_ecdf" for name in DAMAGE_COMPONENTS]
        ].mean(axis=1)
        target["recovery_pressure_outer_median"] = float(
            reference_recovery.median()
        )
        target["damage_exposure_outer_median"] = float(reference_damage.median())
        parts.append(target)
    frozen = (
        pd.concat(parts, ignore_index=True)
        .sort_values("well_id", kind="mergesort")
        .reset_index(drop=True)
    )
    frozen["high_recovery_pressure"] = frozen["recovery_pressure_score"].gt(
        frozen["recovery_pressure_outer_median"]
    )
    frozen["low_damage_exposure"] = frozen["damage_exposure_score"].le(
        frozen["damage_exposure_outer_median"]
    )
    recovery_label = np.where(
        frozen["high_recovery_pressure"],
        "high_recovery_pressure",
        "low_recovery_pressure",
    )
    damage_label = np.where(
        frozen["low_damage_exposure"],
        "low_damage_exposure",
        "high_damage_exposure",
    )
    frozen["regime_cell"] = np.char.add(
        np.char.add(recovery_label.astype(str), "__"),
        damage_label.astype(str),
    )
    frozen["is_primary_target_cell"] = frozen["regime_cell"].eq(TARGET_CELL)
    result = frozen[list(REGIME_FEATURE_COLUMNS)].copy()
    if result["well_id"].duplicated().any() or len(result) != len(audit):
        raise ValueError("regime feature coverage mismatch")
    return result


def build_row_scope_freeze(
    prediction: pd.DataFrame,
    reporting: pd.DataFrame,
    regime_features: pd.DataFrame,
) -> pd.DataFrame:
    frame = prediction[
        [
            "id",
            "well_id",
            "row_idx",
            "suffix_offset",
            "md_since",
            "raw_gr_observed",
        ]
    ].merge(
        reporting[["well_id", "row_idx", "fold"]],
        on=["well_id", "row_idx"],
        how="left",
        validate="one_to_one",
    )
    regime_columns = [
        "well_id",
        "regime_cell",
        "is_primary_target_cell",
    ]
    frame = frame.merge(
        regime_features[regime_columns],
        on="well_id",
        how="left",
        validate="many_to_one",
    )
    eval_rows = frame.groupby("well_id", sort=False)["row_idx"].transform("size")
    expected_offsets = frame.groupby("well_id", sort=False).cumcount().to_numpy(np.int64)
    if not np.array_equal(
        frame["suffix_offset"].to_numpy(np.int64),
        expected_offsets,
    ):
        raise ValueError("suffix_offset is not contiguous within well")
    denominator = (eval_rows - 1).to_numpy(np.float64)
    numerator = frame["suffix_offset"].to_numpy(np.float64)
    progress = np.divide(
        numerator,
        denominator,
        out=np.zeros(len(frame), dtype=np.float64),
        where=denominator > 0.0,
    )
    if np.any(progress < 0.0) or np.any(progress > 1.0):
        raise ValueError("normalized suffix progress is outside [0, 1]")
    quartile_index = np.minimum(np.floor(progress * 4.0).astype(np.int8), 3)
    quartile_labels = np.asarray(
        ["q1_000_025", "q2_025_050", "q3_050_075", "q4_075_100"],
        dtype=object,
    )
    frame["normalized_suffix_progress"] = progress
    frame["suffix_progress_quartile"] = quartile_labels[quartile_index]
    frame["raw_gr_status"] = np.where(
        frame["raw_gr_observed"].astype(bool), "observed", "missing"
    )
    frame["md_since_1000_ft"] = np.where(
        frame["md_since"].ge(1000.0), "at_or_above", "below"
    )
    result = frame[list(ROW_SCOPE_COLUMNS)].copy()
    if (
        result["id"].duplicated().any()
        or result.duplicated(["well_id", "row_idx"]).any()
        or result.isna().any().any()
    ):
        raise ValueError("row-scope freeze coverage mismatch")
    return result


def freeze_target_free_regime(
    prediction: pd.DataFrame,
    audit: pd.DataFrame,
    reporting: pd.DataFrame,
    config: Mapping[str, Any],
    output: Path,
    ledger: OutcomeAccessLedger,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Path]]:
    features = build_fold_safe_regime_features(audit, reporting, config)
    assignment_columns = (
        "well_id",
        "fold",
        "recovery_pressure_score",
        "damage_exposure_score",
        "recovery_pressure_outer_median",
        "damage_exposure_outer_median",
        "high_recovery_pressure",
        "low_damage_exposure",
        "regime_cell",
        "is_primary_target_cell",
    )
    assignment = features[list(assignment_columns)].copy()
    row_scope = build_row_scope_freeze(prediction, reporting, features)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "regime_feature_freeze": output / f"{OUTPUT_PREFIX}_regime_feature_freeze.csv",
        "regime_assignment": output / f"{OUTPUT_PREFIX}_regime_assignment.csv",
        "row_scope_freeze": output / f"{OUTPUT_PREFIX}_row_scope_freeze.csv.gz",
    }
    features.to_csv(paths["regime_feature_freeze"], index=False)
    assignment.to_csv(paths["regime_assignment"], index=False)
    write_deterministic_gzip_csv(row_scope, paths["row_scope_freeze"])
    freeze = {
        "frozen_before_outcome_attachment": True,
        "feature_rows": len(features),
        "assignment_rows": len(assignment),
        "row_scope_rows": len(row_scope),
        "wells": int(features["well_id"].nunique()),
        "folds": sorted(features["fold"].astype(int).unique().tolist()),
        "target_cell": TARGET_CELL,
        "target_cell_wells": int(features["is_primary_target_cell"].sum()),
        "feature_schema_sha256": dataframe_schema_sha(features),
        "feature_logical_content_sha256": dataframe_content_sha(features),
        "assignment_schema_sha256": dataframe_schema_sha(assignment),
        "assignment_logical_content_sha256": dataframe_content_sha(assignment),
        "row_scope_schema_sha256": dataframe_schema_sha(row_scope),
        "row_scope_logical_content_sha256": dataframe_content_sha(row_scope),
        "row_scope_raw_gzip_sha256": sha256_path(paths["row_scope_freeze"]),
        "row_scope_decompressed_sha256": inspect_gzip_csv(
            paths["row_scope_freeze"]
        )["decompressed_sha256"],
    }
    ledger.mark_frozen()
    freeze["outcome_access_ledger_at_freeze"] = ledger.report()
    return features, row_scope, freeze, paths


# %% [markdown]
# ## 6. Late outcome attachment and parent-metric parity


# %%
def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - truth))))


def align_on_id(
    frame: pd.DataFrame,
    source: pd.DataFrame,
    columns: Sequence[str],
    *,
    label: str,
) -> pd.DataFrame:
    lookup = source.copy()
    lookup["id"] = lookup["id"].astype(str)
    if lookup["id"].duplicated().any():
        raise ValueError(f"{label} contains duplicate IDs")
    aligned = lookup.set_index("id").reindex(frame["id"].astype(str))
    if aligned[list(columns)].isna().any().any():
        raise ValueError(f"{label} is missing aligned rows")
    result = frame.copy()
    for column in columns:
        result[column] = aligned[column].to_numpy()
    return result


def build_by_well_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well_id", sort=True):
        truth = group[TRUE_TVT].to_numpy(np.float64)
        candidate = group[PRIMARY_CANDIDATE].to_numpy(np.float64)
        control = group[PRIMARY_CONTROL].to_numpy(np.float64)
        candidate_rmse = rmse(truth, candidate)
        control_rmse = rmse(truth, control)
        rows.append(
            {
                "well_id": str(well),
                "fold": int(group["fold"].iloc[0]),
                "rows": len(group),
                "candidate_rmse": candidate_rmse,
                "control_rmse": control_rmse,
                "improvement_ft": control_rmse - candidate_rmse,
                "delta_rmse_candidate_minus_control": candidate_rmse - control_rmse,
            }
        )
    return pd.DataFrame(rows)


def load_late_outcomes(
    prediction: pd.DataFrame,
    regime_features: pd.DataFrame,
    row_scope: pd.DataFrame,
    preflight: Mapping[str, Any],
    config: Mapping[str, Any],
    ledger: OutcomeAccessLedger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ledger.require_frozen()
    exp226 = _source_spec(config, "exp226_reporting")
    late_columns = list(
        dict.fromkeys(
            [
                *[str(value) for value in exp226["prefreeze_safe_columns"]],
                *[str(value) for value in exp226["postfreeze_truth_columns"]],
            ]
        )
    )
    truth = pd.read_csv(
        preflight["paths"]["exp226_reporting"],
        usecols=late_columns,
        dtype={"well_id": str},
    )
    for column in ("row_idx", "suffix_offset", "fold"):
        truth[column] = pd.to_numeric(truth[column], errors="raise").astype(np.int64)
    truth[TRUE_TVT] = pd.to_numeric(truth[TRUE_TVT], errors="raise").astype(np.float64)
    truth["id"] = (
        truth["well_id"].astype(str)
        + "_"
        + truth["row_idx"].astype(np.int64).astype(str)
    )
    ledger.truth_rows_after_freeze += len(truth)

    exp072 = _source_spec(config, "exp072_control")
    control = pd.read_csv(
        preflight["paths"]["exp072_control"],
        usecols=["id", *[str(value) for value in exp072["reconstruction_columns"]]],
        dtype={"id": str},
    )
    control[PRIMARY_CONTROL] = (
        pd.to_numeric(control["last_known_tvt"], errors="raise")
        + pd.to_numeric(control["likpf_mean_d"], errors="raise")
    )
    ledger.control_rows_after_freeze += len(control)

    frame = align_on_id(
        prediction,
        truth[["id", TRUE_TVT]],
        [TRUE_TVT],
        label="exp226 suffix truth",
    )
    frame = align_on_id(
        frame,
        control[["id", PRIMARY_CONTROL]],
        [PRIMARY_CONTROL],
        label="saved exp072 control",
    )
    frame = frame.merge(
        row_scope,
        on=["id", "well_id", "row_idx", "suffix_offset"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_scope"),
    )
    if frame[list(ROW_SCOPE_COLUMNS[4:])].isna().any().any():
        raise ValueError("frozen row-scope attachment is incomplete")
    numeric = frame[[TRUE_TVT, PRIMARY_CONTROL, PRIMARY_CANDIDATE]].to_numpy(
        np.float64
    )
    if not np.isfinite(numeric).all():
        raise ValueError("late outcome frame contains non-finite values")

    expected_candidate = float(
        get_nested(config, "validation.expected_candidate_rmse_ft")
    )
    expected_control = float(get_nested(config, "validation.expected_control_rmse_ft"))
    expected_delta = float(
        get_nested(config, "validation.expected_candidate_minus_control_rmse_ft")
    )
    tolerance = float(
        get_nested(config, "validation.metric_parity_absolute_tolerance_ft")
    )
    candidate_rmse = rmse(
        frame[TRUE_TVT].to_numpy(np.float64),
        frame[PRIMARY_CANDIDATE].to_numpy(np.float64),
    )
    control_rmse = rmse(
        frame[TRUE_TVT].to_numpy(np.float64),
        frame[PRIMARY_CONTROL].to_numpy(np.float64),
    )
    candidate_minus_control = candidate_rmse - control_rmse
    if (
        abs(candidate_rmse - expected_candidate) > tolerance
        or abs(control_rmse - expected_control) > tolerance
        or abs(candidate_minus_control - expected_delta) > tolerance
    ):
        raise ValueError(
            "exp416 candidate, exp072 control, or pooled delta RMSE parity failed"
        )

    by_well = build_by_well_metrics(frame).merge(
        regime_features[
            [
                "well_id",
                "recovery_pressure_score",
                "damage_exposure_score",
                "regime_cell",
                "is_primary_target_cell",
            ]
        ],
        on="well_id",
        how="left",
        validate="one_to_one",
    )
    saved_by_well = pd.read_csv(
        preflight["paths"]["parent_by_well"],
        dtype={"well_id": str},
    ).sort_values("well_id", kind="mergesort").reset_index(drop=True)
    ledger.by_well_rows_after_freeze += len(saved_by_well)
    by_well_sorted = by_well.sort_values(
        "well_id", kind="mergesort"
    ).reset_index(drop=True)
    if (
        len(saved_by_well) != int(get_nested(config, "validation.expected_wells"))
        or saved_by_well["well_id"].duplicated().any()
        or not np.array_equal(
            saved_by_well["well_id"].astype(str).to_numpy(),
            by_well_sorted["well_id"].astype(str).to_numpy(),
        )
    ):
        raise ValueError("saved exp416 by-well coverage mismatch")
    parity_columns = ("candidate_rmse", "control_rmse", "improvement_ft")
    parity_differences = {
        column: float(
            np.max(
                np.abs(
                    pd.to_numeric(saved_by_well[column], errors="raise").to_numpy(
                        np.float64
                    )
                    - by_well_sorted[column].to_numpy(np.float64)
                )
            )
        )
        for column in parity_columns
    }
    if max(parity_differences.values()) > tolerance:
        raise ValueError("saved exp416 by-well metric parity failed")

    episodes = pd.read_csv(
        preflight["paths"]["parent_episodes"],
        dtype={"episode_id": str, "well_id": str},
    )
    ledger.persistent_episode_rows_after_freeze += len(episodes)
    episodes["rows"] = pd.to_numeric(episodes["rows"], errors="raise").astype(np.int64)
    for column in ("candidate_sse", "control_sse", "sse_reduction_fraction"):
        episodes[column] = pd.to_numeric(episodes[column], errors="raise").astype(
            np.float64
        )
    episodes["improved"] = parse_bool_series(
        episodes["improved"],
        label="exp416 persistent episode improved",
    )
    expected_episode_count = int(
        get_nested(config, "validation.expected_persistent_episodes")
    )
    expected_episode_wells = int(
        get_nested(config, "validation.expected_persistent_episode_wells")
    )
    expected_episode_rows = int(
        get_nested(config, "validation.expected_persistent_episode_rows")
    )
    if (
        len(episodes) != expected_episode_count
        or episodes["episode_id"].duplicated().any()
        or episodes["well_id"].nunique() != expected_episode_wells
        or int(episodes["rows"].sum()) != expected_episode_rows
        or not np.isfinite(
            episodes[
                ["candidate_sse", "control_sse", "sse_reduction_fraction"]
            ].to_numpy(np.float64)
        ).all()
    ):
        raise ValueError("persistent episode coverage mismatch")
    episodes = episodes.merge(
        regime_features[
            ["well_id", "regime_cell", "is_primary_target_cell"]
        ],
        on="well_id",
        how="left",
        validate="many_to_one",
    )
    return frame, by_well_sorted, episodes, {
        "candidate_rmse": candidate_rmse,
        "control_rmse": control_rmse,
        "candidate_minus_control_rmse_ft": candidate_minus_control,
        "candidate_absolute_difference_ft": abs(candidate_rmse - expected_candidate),
        "control_absolute_difference_ft": abs(control_rmse - expected_control),
        "candidate_minus_control_absolute_difference_ft": abs(
            candidate_minus_control - expected_delta
        ),
        "by_well_max_absolute_differences_ft": parity_differences,
        "outcome_access_ledger": ledger.report(),
    }


# %% [markdown]
# ## 7. Association, regime, position, and episode readouts


# %%
def metric_record(
    frame: pd.DataFrame,
    mask: np.ndarray,
    *,
    scope_type: str,
    scope: str,
    fold: int | None = None,
    regime_cell: str | None = None,
) -> dict[str, Any]:
    selected = frame.loc[mask]
    if selected.empty:
        return {
            "scope_type": scope_type,
            "scope": scope,
            "fold": fold,
            "regime_cell": regime_cell,
            "rows": 0,
            "wells": 0,
            "candidate_rmse": np.nan,
            "control_rmse": np.nan,
            "improvement_ft": np.nan,
        }
    truth = selected[TRUE_TVT].to_numpy(np.float64)
    candidate = selected[PRIMARY_CANDIDATE].to_numpy(np.float64)
    control = selected[PRIMARY_CONTROL].to_numpy(np.float64)
    candidate_rmse = rmse(truth, candidate)
    control_rmse = rmse(truth, control)
    return {
        "scope_type": scope_type,
        "scope": scope,
        "fold": fold,
        "regime_cell": regime_cell,
        "rows": len(selected),
        "wells": int(selected["well_id"].nunique()),
        "candidate_rmse": candidate_rmse,
        "control_rmse": control_rmse,
        "improvement_ft": control_rmse - candidate_rmse,
        "candidate_minus_control_rmse_ft": candidate_rmse - control_rmse,
    }


def build_metric_readouts(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    overall_rows = [
        metric_record(
            frame,
            np.ones(len(frame), dtype=bool),
            scope_type="overall",
            scope="overall",
        )
    ]
    for fold in folds:
        overall_rows.append(
            metric_record(
                frame,
                frame["fold"].eq(fold).to_numpy(),
                scope_type="fold",
                scope=f"fold_{fold}",
                fold=fold,
            )
        )

    regime_rows: list[dict[str, Any]] = []
    for cell in sorted(frame["regime_cell"].astype(str).unique()):
        cell_mask = frame["regime_cell"].eq(cell)
        regime_rows.append(
            metric_record(
                frame,
                cell_mask.to_numpy(),
                scope_type="regime",
                scope=cell,
                regime_cell=cell,
            )
        )
        for fold in folds:
            regime_rows.append(
                metric_record(
                    frame,
                    (cell_mask & frame["fold"].eq(fold)).to_numpy(),
                    scope_type="regime_fold",
                    scope=f"{cell}__fold_{fold}",
                    fold=fold,
                    regime_cell=cell,
                )
            )

    position_rows: list[dict[str, Any]] = []
    dimensions = {
        "suffix_progress_quartile": [
            "q1_000_025",
            "q2_025_050",
            "q3_050_075",
            "q4_075_100",
        ],
        "raw_gr_status": ["observed", "missing"],
        "md_since_1000_ft": ["below", "at_or_above"],
    }
    cells: list[str | None] = [None, *sorted(frame["regime_cell"].astype(str).unique())]
    for dimension, levels in dimensions.items():
        for level in levels:
            for cell in cells:
                mask = frame[dimension].eq(level)
                if cell is not None:
                    mask &= frame["regime_cell"].eq(cell)
                position_rows.append(
                    {
                        "dimension": dimension,
                        "level": level,
                        **metric_record(
                            frame,
                            mask.to_numpy(),
                            scope_type="position",
                            scope=f"{dimension}={level}",
                            regime_cell=cell,
                        ),
                    }
                )
    return (
        pd.DataFrame(overall_rows),
        pd.DataFrame(regime_rows),
        pd.DataFrame(position_rows),
    )


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)
    if (
        x_values.size != y_values.size
        or x_values.size < 2
        or not np.isfinite(x_values).all()
        or not np.isfinite(y_values).all()
    ):
        raise ValueError("Spearman inputs must be finite, aligned, and length >= 2")
    x_rank = pd.Series(x_values).rank(method="average").to_numpy(np.float64)
    y_rank = pd.Series(y_values).rank(method="average").to_numpy(np.float64)
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denominator = float(
        np.sqrt(np.square(x_centered).sum() * np.square(y_centered).sum())
    )
    if denominator == 0.0:
        return 0.0
    return float(np.dot(x_centered, y_centered) / denominator)


def within_fold_permutation_p(
    score: Sequence[float],
    gain: Sequence[float],
    fold: Sequence[int],
    *,
    direction: str,
    permutations: int,
    label: str,
) -> dict[str, Any]:
    score_values = np.asarray(score, dtype=np.float64)
    gain_values = np.asarray(gain, dtype=np.float64)
    fold_values = np.asarray(fold, dtype=np.int64)
    observed = spearman_rho(score_values, gain_values)
    score_rank = pd.Series(score_values).rank(method="average").to_numpy(np.float64)
    gain_rank = pd.Series(gain_values).rank(method="average").to_numpy(np.float64)
    score_centered = score_rank - score_rank.mean()
    gain_mean = gain_rank.mean()
    denominator = float(
        np.sqrt(
            np.square(score_centered).sum()
            * np.square(gain_rank - gain_mean).sum()
        )
    )
    if denominator == 0.0:
        raise ValueError("permutation correlation denominator is zero")
    fold_indices = {
        int(value): np.flatnonzero(fold_values == value)
        for value in sorted(np.unique(fold_values))
    }
    rngs = {
        value: np.random.default_rng(
            stable_seed(EXPERIMENT_NAME, label, "fold", value)
        )
        for value in fold_indices
    }
    extreme = 0
    null_min = math.inf
    null_max = -math.inf
    permuted_rank = gain_rank.copy()
    for _ in range(permutations):
        for value, indices in fold_indices.items():
            permuted_rank[indices] = gain_rank[indices][
                rngs[value].permutation(len(indices))
            ]
        rho = float(
            np.dot(score_centered, permuted_rank - gain_mean) / denominator
        )
        null_min = min(null_min, rho)
        null_max = max(null_max, rho)
        if direction == "positive":
            extreme += int(rho >= observed)
        elif direction == "negative":
            extreme += int(rho <= observed)
        else:
            raise ValueError("permutation direction must be positive or negative")
    return {
        "observed_rho": observed,
        "direction": direction,
        "permutations": permutations,
        "extreme_count": extreme,
        "one_sided_p": (extreme + 1.0) / (permutations + 1.0),
        "null_min_rho": null_min,
        "null_max_rho": null_max,
        "fold_seeds": {
            str(value): stable_seed(EXPERIMENT_NAME, label, "fold", value)
            for value in fold_indices
        },
    }


def build_association_readouts(
    regime_features: pd.DataFrame,
    by_well: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    joined = regime_features.merge(
        by_well[["well_id", "improvement_ft"]],
        on="well_id",
        how="left",
        validate="one_to_one",
    )
    folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    diagnostic_rows: list[dict[str, Any]] = []
    for diagnostic in RAW_DIAGNOSTICS:
        diagnostic_rows.append(
            {
                "diagnostic": diagnostic,
                "scope": "pooled",
                "fold": None,
                "rho": spearman_rho(
                    joined[diagnostic],
                    joined["improvement_ft"],
                ),
                "wells": len(joined),
            }
        )
        for fold in folds:
            selected = joined.loc[joined["fold"].eq(fold)]
            diagnostic_rows.append(
                {
                    "diagnostic": diagnostic,
                    "scope": "fold",
                    "fold": fold,
                    "rho": spearman_rho(
                        selected[diagnostic],
                        selected["improvement_ft"],
                    ),
                    "wells": len(selected),
                }
            )

    permutations = int(get_nested(config, "guards.scientific.permutation_count"))
    score_specs = (
        ("recovery_pressure_score", "positive"),
        ("damage_exposure_score", "negative"),
    )
    score_rows: list[dict[str, Any]] = []
    permutation_reports: dict[str, Any] = {}
    for score, direction in score_specs:
        report = within_fold_permutation_p(
            joined[score],
            joined["improvement_ft"],
            joined["fold"],
            direction=direction,
            permutations=permutations,
            label=score,
        )
        permutation_reports[score] = report
        score_rows.append(
            {
                "score": score,
                "scope": "pooled",
                "fold": None,
                "rho": report["observed_rho"],
                "direction": direction,
                "one_sided_permutation_p": report["one_sided_p"],
                "wells": len(joined),
            }
        )
        for fold in folds:
            selected = joined.loc[joined["fold"].eq(fold)]
            score_rows.append(
                {
                    "score": score,
                    "scope": "fold",
                    "fold": fold,
                    "rho": spearman_rho(
                        selected[score],
                        selected["improvement_ft"],
                    ),
                    "direction": direction,
                    "one_sided_permutation_p": np.nan,
                    "wells": len(selected),
                }
            )
    return (
        pd.DataFrame(diagnostic_rows),
        pd.DataFrame(score_rows),
        permutation_reports,
    )


def build_episode_readout(episodes: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = episodes.copy()
    result["sse_reduction"] = result["control_sse"] - result["candidate_sse"]
    target = result.loc[result["is_primary_target_cell"]]
    positive_all = float(result["sse_reduction"].clip(lower=0.0).sum())
    positive_target = float(target["sse_reduction"].clip(lower=0.0).sum())
    target_control_sse = float(target["control_sse"].sum())
    target_candidate_sse = float(target["candidate_sse"].sum())
    target_fraction = (
        1.0 - target_candidate_sse / target_control_sse
        if target_control_sse > 0.0
        else float("-inf")
    )
    summary = {
        "target_cell": TARGET_CELL,
        "episodes": len(target),
        "wells": int(target["well_id"].nunique()),
        "rows": int(target["rows"].sum()),
        "candidate_sse": target_candidate_sse,
        "control_sse": target_control_sse,
        "sse_reduction_fraction": target_fraction,
        "positive_sse_reduction": positive_target,
        "all_improved_episode_positive_sse_reduction": positive_all,
        "share_of_positive_episode_sse_reduction": (
            positive_target / positive_all if positive_all > 0.0 else 0.0
        ),
    }
    return result, summary


# %% [markdown]
# ## 8. Technical and scientific attribution gates


# %%
def _single_metric_row(
    metrics: pd.DataFrame,
    *,
    scope_type: str,
    regime_cell: str | None = None,
    fold: int | None = None,
) -> pd.Series:
    mask = metrics["scope_type"].eq(scope_type)
    if regime_cell is not None:
        mask &= metrics["regime_cell"].eq(regime_cell)
    if fold is not None:
        mask &= metrics["fold"].eq(fold)
    selected = metrics.loc[mask]
    if len(selected) != 1:
        raise ValueError(
            f"expected one metric row: type={scope_type}, cell={regime_cell}, fold={fold}"
        )
    return selected.iloc[0]


def evaluate_attribution_gate(
    preflight: Mapping[str, Any],
    freeze: Mapping[str, Any],
    parity: Mapping[str, Any],
    frame: pd.DataFrame,
    by_well: pd.DataFrame,
    episodes: pd.DataFrame,
    overall_metrics: pd.DataFrame,
    regime_metrics: pd.DataFrame,
    score_associations: pd.DataFrame,
    permutation_reports: Mapping[str, Any],
    episode_summary: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    technical_config = dict(get_nested(config, "guards.technical") or {})
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_folds = [
        int(value) for value in get_nested(config, "validation.expected_folds")
    ]
    before_freeze = freeze["outcome_access_ledger_at_freeze"][
        "outcome_before_freeze"
    ]
    execution = execution_counts(config)
    expected_execution = {
        "saved_output_readout_contracts": 1,
        "new_prediction_rows": 0,
        "scientific_pf_variants": 0,
        "candidate_pf_well_runs": 0,
        "parent_pf_control_reruns": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_well_runs": 0,
        "beam_well_runs": 0,
        "gpu_runs": 0,
        "reporting_folds": 5,
    }
    metric_tolerance = float(
        technical_config["require_candidate_control_metric_parity_atol_ft"]
    )
    technical_checks = {
        "all_input_sha_matches": bool(preflight["all_input_sha_matches"]),
        "parent_terminal_fail_unchanged": bool(
            preflight["parent_terminal_contract"]["passed"]
            and not preflight["parent_terminal_contract"]["parent_reclassified"]
        ),
        "source_kernel_version": (
            int(preflight["source_kernel"]["version"])
            == int(technical_config["require_source_kernel_version"])
        ),
        "source_kernel_id_no": (
            int(preflight["source_kernel"]["id_no"])
            == int(technical_config["require_source_kernel_id_no"])
        ),
        "rows": len(frame) == expected_rows,
        "wells": frame["well_id"].nunique() == expected_wells,
        "folds": sorted(frame["fold"].astype(int).unique().tolist())
        == expected_folds,
        "unique_identity": (
            not frame["id"].duplicated().any()
            and not frame.duplicated(["well_id", "row_idx"]).any()
        ),
        "finite_prediction_and_outcome_coverage": bool(
            np.isfinite(
                frame[[TRUE_TVT, PRIMARY_CONTROL, PRIMARY_CANDIDATE]].to_numpy(
                    np.float64
                )
            ).mean()
            == float(technical_config["require_finite_diagnostic_and_prediction_coverage"])
        ),
        "pooled_metric_parity": (
            float(parity["candidate_absolute_difference_ft"]) <= metric_tolerance
            and float(parity["control_absolute_difference_ft"]) <= metric_tolerance
            and float(parity["candidate_minus_control_absolute_difference_ft"])
            <= metric_tolerance
        ),
        "by_well_metric_parity": max(
            float(value)
            for value in parity["by_well_max_absolute_differences_ft"].values()
        )
        <= metric_tolerance,
        "well_audit_rows": int(
            freeze["outcome_access_ledger_at_freeze"]["prefreeze_safe_rows"][
                "well_audit"
            ]
        )
        == int(technical_config["require_well_audit_rows"]),
        "by_well_rows": len(by_well)
        == int(technical_config["require_by_well_metric_rows"]),
        "persistent_episode_count": len(episodes)
        == int(technical_config["require_persistent_episode_count"]),
        "persistent_episode_wells": episodes["well_id"].nunique()
        == int(technical_config["require_persistent_episode_wells"]),
        "persistent_episode_rows": int(episodes["rows"].sum())
        == int(technical_config["require_persistent_episode_rows"]),
        "outcome_rows_before_freeze": all(
            int(value) == int(technical_config["maximum_outcome_rows_opened_before_regime_freeze"])
            for value in before_freeze.values()
        ),
        "feature_and_scope_sha": all(
            len(str(freeze[key])) == 64
            for key in (
                "feature_schema_sha256",
                "feature_logical_content_sha256",
                "assignment_schema_sha256",
                "assignment_logical_content_sha256",
                "row_scope_schema_sha256",
                "row_scope_logical_content_sha256",
            )
        ),
        "execution_count_match": execution == expected_execution,
    }
    technical = {
        "passed": bool(all(technical_checks.values())),
        "checks": technical_checks,
        "execution_counts": execution,
        "expected_execution_counts": expected_execution,
        "outcome_rows_before_freeze": before_freeze,
        "feature_freeze": dict(freeze),
        "parity": dict(parity),
    }

    scientific_config = dict(get_nested(config, "guards.scientific") or {})
    recovery_pooled = score_associations.loc[
        score_associations["score"].eq("recovery_pressure_score")
        & score_associations["scope"].eq("pooled")
    ].iloc[0]
    damage_pooled = score_associations.loc[
        score_associations["score"].eq("damage_exposure_score")
        & score_associations["scope"].eq("pooled")
    ].iloc[0]
    recovery_fold_rows = score_associations.loc[
        score_associations["score"].eq("recovery_pressure_score")
        & score_associations["scope"].eq("fold")
    ]
    damage_fold_rows = score_associations.loc[
        score_associations["score"].eq("damage_exposure_score")
        & score_associations["scope"].eq("fold")
    ]
    recovery_positive_folds = int(recovery_fold_rows["rho"].gt(0.0).sum())
    damage_negative_folds = int(damage_fold_rows["rho"].lt(0.0).sum())

    target_overall = _single_metric_row(
        regime_metrics,
        scope_type="regime",
        regime_cell=TARGET_CELL,
    )
    target_fold_rows = regime_metrics.loc[
        regime_metrics["scope_type"].eq("regime_fold")
        & regime_metrics["regime_cell"].eq(TARGET_CELL)
    ]
    target_improved_folds = int(target_fold_rows["improvement_ft"].gt(0.0).sum())
    target_wells = by_well.loc[by_well["is_primary_target_cell"]]
    rest_wells = by_well.loc[~by_well["is_primary_target_cell"]]
    target_minus_rest = float(
        target_wells["improvement_ft"].mean() - rest_wells["improvement_ft"].mean()
    )
    target_improved_well_fraction = float(
        target_wells["improvement_ft"].gt(0.0).mean()
    )
    recovery = {
        "pooled_rho": float(recovery_pooled["rho"]),
        "minimum_pooled_rho": float(
            scientific_config["recovery_pressure_minimum_pooled_spearman_rho"]
        ),
        "positive_folds": recovery_positive_folds,
        "minimum_positive_folds": int(
            scientific_config["recovery_pressure_minimum_positive_folds"]
        ),
        "one_sided_permutation_p": float(
            permutation_reports["recovery_pressure_score"]["one_sided_p"]
        ),
        "maximum_one_sided_permutation_p": float(
            scientific_config[
                "recovery_pressure_maximum_one_sided_permutation_p"
            ]
        ),
    }
    recovery["passed"] = bool(
        recovery["pooled_rho"] >= recovery["minimum_pooled_rho"]
        and recovery["positive_folds"] >= recovery["minimum_positive_folds"]
        and recovery["one_sided_permutation_p"]
        <= recovery["maximum_one_sided_permutation_p"]
    )
    damage = {
        "pooled_rho": float(damage_pooled["rho"]),
        "maximum_pooled_rho": float(
            scientific_config["damage_exposure_maximum_pooled_spearman_rho"]
        ),
        "negative_folds": damage_negative_folds,
        "minimum_negative_folds": int(
            scientific_config["damage_exposure_minimum_negative_folds"]
        ),
        "one_sided_permutation_p": float(
            permutation_reports["damage_exposure_score"]["one_sided_p"]
        ),
        "maximum_one_sided_permutation_p": float(
            scientific_config["damage_exposure_maximum_one_sided_permutation_p"]
        ),
    }
    damage["passed"] = bool(
        damage["pooled_rho"] <= damage["maximum_pooled_rho"]
        and damage["negative_folds"] >= damage["minimum_negative_folds"]
        and damage["one_sided_permutation_p"]
        <= damage["maximum_one_sided_permutation_p"]
    )
    target_row = {
        "rmse_gain_ft": float(target_overall["improvement_ft"]),
        "minimum_rmse_gain_ft": float(
            scientific_config["minimum_target_cell_row_rmse_gain_ft"]
        ),
        "improved_folds": target_improved_folds,
        "minimum_improved_folds": int(
            scientific_config["minimum_target_cell_improved_folds"]
        ),
    }
    target_row["passed"] = bool(
        target_row["rmse_gain_ft"] >= target_row["minimum_rmse_gain_ft"]
        and target_row["improved_folds"] >= target_row["minimum_improved_folds"]
    )
    target_well = {
        "target_wells": len(target_wells),
        "rest_wells": len(rest_wells),
        "target_minus_rest_equal_well_mean_gain_ft": target_minus_rest,
        "minimum_target_minus_rest_equal_well_mean_gain_ft": float(
            scientific_config[
                "minimum_target_cell_minus_rest_equal_well_mean_gain_ft"
            ]
        ),
        "target_improved_well_fraction": target_improved_well_fraction,
        "minimum_target_improved_well_fraction": float(
            scientific_config["minimum_target_cell_improved_well_fraction"]
        ),
    }
    target_well["passed"] = bool(
        target_well["target_minus_rest_equal_well_mean_gain_ft"]
        >= target_well["minimum_target_minus_rest_equal_well_mean_gain_ft"]
        and target_well["target_improved_well_fraction"]
        >= target_well["minimum_target_improved_well_fraction"]
    )
    episode = {
        **dict(episode_summary),
        "minimum_episodes": int(
            scientific_config["minimum_target_cell_persistent_episodes"]
        ),
        "minimum_wells": int(
            scientific_config["minimum_target_cell_persistent_episode_wells"]
        ),
        "minimum_sse_reduction_fraction": float(
            scientific_config[
                "minimum_target_cell_persistent_episode_sse_reduction_fraction"
            ]
        ),
        "minimum_share_of_positive_episode_sse_reduction": float(
            scientific_config[
                "minimum_target_cell_share_of_positive_episode_sse_reduction"
            ]
        ),
    }
    episode["passed"] = bool(
        episode["episodes"] >= episode["minimum_episodes"]
        and episode["wells"] >= episode["minimum_wells"]
        and episode["sse_reduction_fraction"]
        >= episode["minimum_sse_reduction_fraction"]
        and episode["share_of_positive_episode_sse_reduction"]
        >= episode["minimum_share_of_positive_episode_sse_reduction"]
    )
    scientific = {
        "passed": bool(
            recovery["passed"]
            and damage["passed"]
            and target_row["passed"]
            and target_well["passed"]
            and episode["passed"]
        ),
        "recovery_pressure": recovery,
        "damage_exposure": damage,
        "target_cell_row_weighted": target_row,
        "target_cell_equal_well": target_well,
        "target_cell_persistent_episode": episode,
    }
    passed = bool(technical["passed"] and scientific["passed"])
    return {
        "experiment": EXPERIMENT_NAME,
        "passed": passed,
        "decision": (
            str(get_nested(config, "guards.decision.pass_action"))
            if passed
            else str(get_nested(config, "guards.decision.fail_action"))
        ),
        "technical_gate": technical,
        "scientific_gate": scientific,
        "parent_decision": preflight["parent_terminal_contract"]["decision"],
        "parent_decision_remains_unchanged": True,
        "policy_ready": False,
        "pass_interpretation": (
            "association evidence only; a separately preregistered policy experiment "
            "is required"
        ),
        "failure_action": (
            "close_without_score_weight_transform_threshold_cell_roughening_"
            "multiplier_position_rate_ess_policy_or_same_oof_rescue"
        ),
        "overall_metrics": overall_metrics.to_dict(orient="records"),
    }


# %% [markdown]
# ## 9. Generated artifacts and execution orchestration


# %%
def artifact_report(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": sha256_path(path),
    }
    if path.suffix == ".gz":
        report["decompressed_sha256"] = inspect_gzip_csv(path)[
            "decompressed_sha256"
        ]
    return report


def build_artifact_manifest(paths: Mapping[str, Path]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"name": name, **artifact_report(path)} for name, path in paths.items()]
    ).sort_values("name", kind="mergesort")


def run_full_experiment(
    config: Mapping[str, Any],
    *,
    require_run_approval: bool = True,
) -> dict[str, Any]:
    contract = validate_scientific_contract(
        config,
        require_run_approval=require_run_approval,
    )
    started = time.time()
    preflight = preflight_saved_inputs(config)
    ledger = OutcomeAccessLedger()
    prediction, audit, reporting = load_target_free_inputs(
        preflight,
        config,
        ledger,
    )
    output = artifact_dir()
    regime_features, row_scope, freeze, paths = freeze_target_free_regime(
        prediction,
        audit,
        reporting,
        config,
        output,
        ledger,
    )
    frame, by_well, episodes, parity = load_late_outcomes(
        prediction,
        regime_features,
        row_scope,
        preflight,
        config,
        ledger,
    )
    overall_metrics, regime_metrics, position_metrics = build_metric_readouts(
        frame,
        config,
    )
    diagnostic_associations, score_associations, permutation_reports = (
        build_association_readouts(regime_features, by_well, config)
    )
    episode_readout, episode_summary = build_episode_readout(episodes)
    gate = evaluate_attribution_gate(
        preflight,
        freeze,
        parity,
        frame,
        by_well,
        episodes,
        overall_metrics,
        regime_metrics,
        score_associations,
        permutation_reports,
        episode_summary,
        config,
    )
    paths.update(
        {
            "overall_fold_metrics": output
            / f"{OUTPUT_PREFIX}_overall_fold_metrics.csv",
            "regime_metrics": output / f"{OUTPUT_PREFIX}_regime_metrics.csv",
            "position_scope_metrics": output
            / f"{OUTPUT_PREFIX}_position_scope_metrics.csv",
            "by_well_metrics": output / f"{OUTPUT_PREFIX}_by_well_metrics.csv",
            "individual_diagnostic_associations": output
            / f"{OUTPUT_PREFIX}_individual_diagnostic_associations.csv",
            "score_associations": output
            / f"{OUTPUT_PREFIX}_score_associations.csv",
            "persistent_episode_regime_readout": output
            / f"{OUTPUT_PREFIX}_persistent_episode_regime_readout.csv",
            "permutation_report": output
            / f"{OUTPUT_PREFIX}_permutation_report.json",
            "scientific_contract": output
            / f"{OUTPUT_PREFIX}_scientific_contract.json",
            "attribution_gate": output / f"{OUTPUT_PREFIX}_attribution_gate.json",
        }
    )
    overall_metrics.to_csv(paths["overall_fold_metrics"], index=False)
    regime_metrics.to_csv(paths["regime_metrics"], index=False)
    position_metrics.to_csv(paths["position_scope_metrics"], index=False)
    by_well.to_csv(paths["by_well_metrics"], index=False)
    diagnostic_associations.to_csv(
        paths["individual_diagnostic_associations"], index=False
    )
    score_associations.to_csv(paths["score_associations"], index=False)
    episode_readout.to_csv(
        paths["persistent_episode_regime_readout"], index=False
    )
    write_json(paths["permutation_report"], permutation_reports)
    write_json(paths["scientific_contract"], contract)
    write_json(paths["attribution_gate"], gate)
    artifact_manifest = build_artifact_manifest(paths)
    artifact_manifest_path = output / f"{OUTPUT_PREFIX}_artifact_manifest.csv"
    artifact_manifest.to_csv(artifact_manifest_path, index=False)
    artifact_manifest_sha = sha256_path(artifact_manifest_path)
    status = (
        "target_free_regime_attribution_supported_separate_policy_experiment_required"
        if gate["passed"]
        else "no_reproducible_target_free_regime_close_attribution_branch"
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "route": "pf_beam",
        "runtime_seconds": time.time() - started,
        "peak_rss_gb": maximum_rss_gb(),
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "source_kernel": preflight["source_kernel"],
        "parent_terminal_contract": preflight["parent_terminal_contract"],
        "scientific_contract_sha256": contract["scientific_contract_sha256"],
        "feature_freeze": freeze,
        "outcome_parity": parity,
        "permutation_reports": permutation_reports,
        "episode_summary": episode_summary,
        "gate": gate,
        "execution_counts": execution_counts(config),
        "artifact_manifest_sha256": artifact_manifest_sha,
        "runtime_versions": runtime_versions(),
        "new_prediction_sha256": None,
        "model_sha256": None,
        "submission_sha256": None,
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    summary_path = output / f"{OUTPUT_PREFIX}_summary.json"
    write_json(summary_path, summary)
    overall = overall_metrics.loc[overall_metrics["scope"].eq("overall")].iloc[0]
    metrics_json = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "route": "pf_beam",
        "parent": str(get_nested(config, "lineage.parent")),
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "metric": "saved_output_target_free_failure_regime_attribution",
        "candidate_rmse": float(overall["candidate_rmse"]),
        "control_rmse": float(overall["control_rmse"]),
        "improvement_ft": float(overall["improvement_ft"]),
        "gate": gate,
        "feature_freeze_sha256": freeze["feature_logical_content_sha256"],
        "row_scope_freeze_sha256": freeze["row_scope_logical_content_sha256"],
        "artifact_manifest_sha256": artifact_manifest_sha,
        "new_prediction_rows": 0,
        "pf_well_runs": 0,
        "model_configs": 0,
        "boosters": 0,
        "gpu_runs": 0,
        "notes": (
            "Saved-output attribution only. exp416 remains terminally failed; "
            "no PF, prediction, model, inference, or submission is produced."
        ),
    }
    write_json(metrics_output_path(), metrics_json)
    print(overall_metrics.to_string(index=False))
    print(score_associations.to_string(index=False))
    print(json.dumps(to_jsonable(gate), indent=2, sort_keys=True))
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 10. Setup and configuration preview


# %%
CONFIG = load_experiment_config()
SCIENTIFIC_CONTRACT = validate_scientific_contract(CONFIG)
print("Experiment:", EXPERIMENT_NAME)
print("Route:", get_nested(CONFIG, "experiment.route"))
print("Parent:", get_nested(CONFIG, "lineage.parent"))
print("Implementation status:", get_nested(CONFIG, "experiment.status"))
print("Primary target cell:", TARGET_CELL)
print("Execution counts:", json.dumps(execution_counts(CONFIG), sort_keys=True))
print(
    "Scientific contract SHA:",
    SCIENTIFIC_CONTRACT["scientific_contract_sha256"],
)
print(
    "Canonical notebook adoption approved:",
    get_nested(CONFIG, "execution.canonical_notebook_adoption_approved"),
)
print("Kaggle audit run approved:", get_nested(CONFIG, "execution.audit_run_approved"))


# %% [markdown]
# ## 11. Run the approved Kaggle CPU readout
#
# The compact candidate is implementation-complete but is not the canonical
# notebook and is not approved for packaging or execution yet. The call remains
# fail-closed until all four explicit approval flags are recorded in config.


# %%
if EXECUTE_NOTEBOOK:
    if bool(get_nested(CONFIG, "execution.audit_run_approved")):
        SUMMARY = run_full_experiment(CONFIG, require_run_approval=True)
    else:
        print(
            "exp422 implementation preview only: canonical notebook adoption, "
            "Kaggle package/push, and audit execution remain unapproved."
        )
