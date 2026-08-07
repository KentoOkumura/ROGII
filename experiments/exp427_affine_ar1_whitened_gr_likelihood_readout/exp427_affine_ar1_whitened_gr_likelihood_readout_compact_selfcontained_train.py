# %% [markdown]
# # exp427 affine + AR(1) whitened GR likelihood readout — train
#
# This compact self-contained notebook implements the design-frozen, train-side
# Stage 0 diagnostic. It compares a fixed identity/affine × iid/AR(1) Gaussian
# factorial on the exp280 13-shift/512-row surface. All target-free scores,
# eligibility decisions, fold priors, controls, and content hashes are frozen
# before unknown-suffix TVT or hidden-like roles are attached.
#
# The notebook produces no corrected prediction, HMM/PF/Beam decode, model, or
# submission. Version 2 completed the Stage 0 CPU audit and the checked-in
# post-run contract now disables any repeat execution.

# %% [markdown]
# ## Contents
# 1. Imports and immutable experiment constants
# 2. Notebook-safe configuration, path, and SHA helpers
# 3. Frozen scientific/execution contract and truth-access ledger
# 4. Target-free input loaders and immutable dependency checks
# 5. Known-prefix affine posterior and fold-safe AR(1) prior
# 6. Raw-finite block predictive likelihood and 2×2 factorial
# 7. Saved exp280 alignment and target-free bundle freeze
# 8. Late truth attachment and rank readouts
# 9. Scope metrics and technical/scientific AND gates
# 10. Kaggle CPU orchestration and generated artifacts
# 11. Setup and configuration preview
# 12. Fail-closed Stage 0 entrypoint

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import resource
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

EXPERIMENT_NAME = "exp427_affine_ar1_whitened_gr_likelihood_readout"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
EXPECTED_SHIFTS = np.asarray(
    [-80.0, -40.0, -20.0, -10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0],
    dtype=np.float64,
)
FACTORIAL_VARIANTS = (
    "identity_iid_matched",
    "affine_iid",
    "identity_ar1",
    "affine_ar1",
)
PRIMARY_VARIANT = "affine_ar1"
MATCHED_CONTROL = "identity_iid_matched"
SAVED_CONTROL = "saved_exp280"
NEGATIVE_CONTROL = "affine_ar1_shuffled"


def in_notebook_runtime() -> bool:
    try:
        shell = get_ipython()  # type: ignore[name-defined]  # noqa: F821
    except NameError:
        return False
    return shell is not None


# %% [markdown]
# ## 2. Notebook-safe configuration, path, and SHA helpers
#
# `Path.cwd()` is the anchor so the Jupytext source remains safe after conversion
# to a Kaggle notebook, which does not provide a script-file path.

# %%
def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def get_nested(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def project_root() -> Path:
    candidates = [Path.cwd(), *Path.cwd().parents, KAGGLE_WORKING_ROOT]
    for candidate in candidates:
        if (candidate / "project.yml").exists():
            return candidate
    return Path.cwd()


def experiment_dir() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        restored = KAGGLE_WORKING_ROOT / "experiments" / EXPERIMENT_NAME
        if restored.exists():
            return restored
    return project_root() / "experiments" / EXPERIMENT_NAME


def load_config() -> dict[str, Any]:
    candidates = (
        Path.cwd() / "config.yaml",
        experiment_dir() / "config.yaml",
        KAGGLE_WORKING_ROOT / "config.yaml",
    )
    for path in candidates:
        if path.exists():
            config = read_yaml(path)
            if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
                return config
    raise FileNotFoundError(f"{EXPERIMENT_NAME} config.yaml was not restored")


def artifact_dir() -> Path:
    output = (
        KAGGLE_WORKING_ROOT / "artifacts"
        if KAGGLE_WORKING_ROOT.exists()
        else experiment_dir() / "artifacts"
    )
    output.mkdir(parents=True, exist_ok=True)
    return output


def metrics_output_path() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        return KAGGLE_WORKING_ROOT / "metrics.json"
    return experiment_dir() / "metrics.json"


def train_data_dir(config: Mapping[str, Any]) -> Path:
    expected = int(get_nested(config, "validation.expected_wells"))
    local = project_root() / str(get_nested(config, "data.train_dir", "data/raw/train"))
    if len(list(local.glob("*__horizontal_well.csv"))) == expected:
        return local
    fixed = (
        KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction" / "train",
        KAGGLE_INPUT_ROOT
        / "competitions"
        / "rogii-wellbore-geology-prediction"
        / "train",
    )
    for candidate in fixed:
        if len(list(candidate.glob("*__horizontal_well.csv"))) == expected:
            return candidate
    if KAGGLE_INPUT_ROOT.exists():
        for candidate in sorted(KAGGLE_INPUT_ROOT.glob("**/train")):
            if len(list(candidate.glob("*__horizontal_well.csv"))) == expected:
                return candidate
    return local


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


def mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        to_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def logical_manifest_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): logical_manifest_payload(item)
            for key, item in value.items()
            if str(key) != "path" and not str(key).endswith("_path")
        }
    if isinstance(value, (list, tuple)):
        return [logical_manifest_payload(item) for item in value]
    return value


def dataframe_content_sha(
    frame: pd.DataFrame, columns: Sequence[str] | None = None
) -> str:
    chosen = list(frame.columns) if columns is None else list(columns)
    digest = hashlib.sha256()
    for column in chosen:
        digest.update(str(column).encode())
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
    return mapping_sha256(
        {
            "columns": list(frame.columns),
            "dtypes": {column: str(frame[column].dtype) for column in frame.columns},
        }
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    )


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
        "schema_sha256": dataframe_schema_sha(frame),
    }


def write_csv(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return {
        "path": str(path),
        "rows": len(frame),
        "raw_sha256": sha256_path(path),
        "content_sha256": dataframe_content_sha(frame),
        "schema_sha256": dataframe_schema_sha(frame),
    }


def _expand_pattern(pattern: str) -> list[Path]:
    path = Path(pattern)
    if path.is_absolute() and not any(char in pattern for char in "*?["):
        return [path] if path.is_file() else []
    roots = [project_root(), Path.cwd()]
    if KAGGLE_INPUT_ROOT.exists():
        roots.append(KAGGLE_INPUT_ROOT)
    matches: list[Path] = []
    for root in roots:
        if any(char in pattern for char in "*?["):
            relative = pattern
            if pattern.startswith("**/"):
                relative = pattern[3:]
                matches.extend(sorted(root.rglob(relative)))
            else:
                matches.extend(sorted(root.glob(relative)))
        else:
            candidate = root / path
            if candidate.is_file():
                matches.append(candidate)
    unique: dict[str, Path] = {}
    for match in matches:
        if match.is_file():
            unique[str(match.resolve())] = match
    return [unique[key] for key in sorted(unique)]


def resolve_artifact(
    filename: str,
    *,
    candidates: Iterable[str] = (),
    patterns: Iterable[str] = (),
) -> Path:
    checked: list[str] = []
    for raw in candidates:
        checked.append(str(raw))
        matches = _expand_pattern(str(raw))
        if matches:
            return matches[0]
    for raw in patterns:
        checked.append(str(raw))
        matches = _expand_pattern(str(raw))
        if matches:
            exact = [path for path in matches if path.name == filename]
            return (exact or matches)[0]
    if KAGGLE_INPUT_ROOT.exists():
        matches = sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}"))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"could not resolve {filename}; checked={checked}")


def stable_uint64(*parts: Any) -> int:
    payload = "|".join(str(part) for part in parts).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def stable_score_permutation(
    well_id: str, fold: int, block_id: int, size: int, *, seed: int = 42
) -> np.ndarray:
    rng = np.random.default_rng(
        stable_uint64(EXPERIMENT_NAME, seed, fold, well_id, block_id)
    )
    order = rng.permutation(int(size))
    if size > 1 and np.array_equal(order, np.arange(size)):
        order = np.roll(order, 1)
    return order


def rank_descending(scores: Sequence[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("ranking requires one finite score per candidate")
    order = np.argsort(-values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int16)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.int16)
    return ranks


def peak_rss_gb() -> float:
    usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0 if os.name != "posix" else 1024.0 * 1024.0
    return usage / divisor


# %% [markdown]
# ## 3. Frozen scientific/execution contract and truth-access ledger

# %%
def scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = {
        "experiment": EXPERIMENT_NAME,
        "route": get_nested(config, "experiment.route"),
        "parent": get_nested(config, "lineage.parent"),
        "truth_attached": False,
        "validation": {
            "strategy": get_nested(config, "validation.strategy"),
            "folds": get_nested(config, "validation.expected_folds"),
            "rows": get_nested(config, "validation.expected_rows"),
            "wells": get_nested(config, "validation.expected_wells"),
            "blocks": get_nested(config, "validation.expected_blocks"),
            "technical_gate": get_nested(config, "validation.technical_gate"),
            "scientific_gate": get_nested(config, "validation.scientific_gate"),
        },
        "factorial": get_nested(config, "model.factorial"),
        "affine": get_nested(config, "model.affine"),
        "sigma": get_nested(config, "model.sigma"),
        "ar1": get_nested(config, "model.ar1"),
        "predictive_density": get_nested(config, "model.predictive_density"),
        "shift_bank_ft": get_nested(config, "audit.shift_bank_ft"),
        "block_rows": get_nested(config, "audit.block_rows"),
        "block_policy": get_nested(config, "audit.block_policy"),
        "missing_policy": get_nested(config, "audit.missing_policy"),
        "tie_policy": get_nested(config, "audit.tie_policy"),
        "negative_control": get_nested(config, "audit.negative_control"),
        "forbidden": get_nested(config, "model.forbidden"),
        "execution_contract": get_nested(config, "execution_contract"),
    }
    contract["scientific_contract_sha256"] = mapping_sha256(contract)
    return contract


def validate_scientific_contract(
    config: Mapping[str, Any], *, require_run_approval: bool = False
) -> dict[str, Any]:
    failures: list[str] = []
    if get_nested(config, "experiment.name") != EXPERIMENT_NAME:
        failures.append("experiment.name")
    if get_nested(config, "experiment.route") != "pf_beam":
        failures.append("experiment.route")
    if not bool(get_nested(config, "implementation.enabled")):
        failures.append("implementation.enabled")
    if get_nested(config, "implementation.scope") != "stage_0_rank_audit_only":
        failures.append("implementation.scope")
    if not bool(get_nested(config, "implementation.stage_0_implemented")):
        failures.append("implementation.stage_0_implemented")
    if [float(value) for value in get_nested(config, "audit.shift_bank_ft", [])] != list(
        EXPECTED_SHIFTS
    ):
        failures.append("audit.shift_bank_ft")
    if int(get_nested(config, "audit.block_rows", 0)) != 512:
        failures.append("audit.block_rows")
    if get_nested(config, "audit.tie_policy") != "config_shift_bank_order":
        failures.append("audit.tie_policy")
    if tuple(get_nested(config, "model.factorial.variants", [])) != FACTORIAL_VARIANTS:
        failures.append("model.factorial.variants")
    if get_nested(config, "model.factorial.primary") != PRIMARY_VARIANT:
        failures.append("model.factorial.primary")
    if get_nested(config, "model.factorial.matched_control") != MATCHED_CONTROL:
        failures.append("model.factorial.matched_control")
    if get_nested(config, "model.ar1.yule_walker_formula") != (
        "sum_lagged_pair_products_div_sum_lagged_left_squares"
    ):
        failures.append("model.ar1.yule_walker_formula")
    expected_counts = {
        "scientific_primary_scores": 1,
        "diagnostic_ablation_scores": 2,
        "matched_control_scores": 1,
        "saved_control_scores": 1,
        "reporting_folds": 5,
        "hmm_well_runs": 0,
        "pf_runs": 0,
        "beam_runs": 0,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "gpu_runs": 0,
    }
    if get_nested(config, "execution_contract.stage_0") != expected_counts:
        failures.append("execution_contract.stage_0")
    forbidden_true = (
        "execution.run_hmm",
        "execution.run_pf",
        "execution.run_beam",
        "execution.run_inference",
        "execution.create_prediction",
        "execution.create_submission",
        "inference.enabled",
        "inference.create_submission",
    )
    failures.extend(key for key in forbidden_true if bool(get_nested(config, key)))
    if failures:
        raise ValueError(f"exp427 scientific contract changed: {sorted(failures)}")
    if require_run_approval:
        approvals = (
            "execution.implementation_approved",
            "execution.kaggle_package_approved",
            "execution.kaggle_push_approved",
            "execution.run_stage_0",
            "runtime.kaggle.train_run_on_push",
        )
        missing = [key for key in approvals if not bool(get_nested(config, key))]
        if missing:
            raise PermissionError(f"exp427 Kaggle Stage 0 is not approved: {missing}")
    return scientific_contract(config)


class TruthAccessLedger:
    def __init__(self) -> None:
        self.frozen = False
        self.truth_rows_before_freeze = 0
        self.truth_rows_after_freeze = 0
        self.hidden_rows_before_freeze = 0
        self.hidden_rows_after_freeze = 0
        self.frozen_bundle_sha256 = ""

    def mark_frozen(self, bundle_sha256: str) -> None:
        if len(str(bundle_sha256)) != 64:
            raise ValueError("target-free bundle freeze requires a SHA256")
        if self.truth_rows_before_freeze or self.hidden_rows_before_freeze:
            raise RuntimeError("late-only information was accessed before freeze")
        self.frozen = True
        self.frozen_bundle_sha256 = str(bundle_sha256)

    def require_frozen(self) -> None:
        if not self.frozen:
            raise RuntimeError("late truth/role access requires a frozen target-free bundle")

    def register_truth(self, rows: int) -> None:
        if not self.frozen:
            self.truth_rows_before_freeze += int(rows)
            raise RuntimeError("truth access attempted before target-free freeze")
        self.truth_rows_after_freeze += int(rows)

    def register_hidden_roles(self, rows: int) -> None:
        if not self.frozen:
            self.hidden_rows_before_freeze += int(rows)
            raise RuntimeError("hidden-like role access attempted before target-free freeze")
        self.hidden_rows_after_freeze += int(rows)

    def report(self) -> dict[str, Any]:
        return {
            "frozen": self.frozen,
            "frozen_bundle_sha256": self.frozen_bundle_sha256,
            "truth_rows_before_freeze": self.truth_rows_before_freeze,
            "truth_rows_after_freeze": self.truth_rows_after_freeze,
            "hidden_rows_before_freeze": self.hidden_rows_before_freeze,
            "hidden_rows_after_freeze": self.hidden_rows_after_freeze,
        }


# %% [markdown]
# ## 4. Target-free input loaders and immutable dependency checks

# %%
def validate_raw_well_identity(
    config: Mapping[str, Any], raw_dir: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizontal_path in sorted(raw_dir.glob("*__horizontal_well.csv")):
        well = horizontal_path.name.removesuffix("__horizontal_well.csv")
        typewell_path = raw_dir / f"{well}__typewell.csv"
        if not typewell_path.is_file():
            raise FileNotFoundError(typewell_path)
        rows.append(
            {
                "well_id": well,
                "horizontal_raw_sha256": sha256_path(horizontal_path),
                "typewell_raw_sha256": sha256_path(typewell_path),
            }
        )
    frame = pd.DataFrame(rows).sort_values("well_id", kind="mergesort").reset_index(
        drop=True
    )
    content_sha = dataframe_content_sha(
        frame, ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"]
    )
    if len(frame) != int(get_nested(config, "validation.expected_wells")):
        raise ValueError("raw train well count changed")
    if content_sha != str(get_nested(config, "data.expected_raw_well_identity_sha256")):
        raise ValueError("raw train well-file identity SHA changed")
    return frame, {
        "name": "raw_train_well_file_identity",
        "path": str(raw_dir),
        "wells": len(frame),
        "content_sha256": content_sha,
    }


def load_exp226_safe(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    spec = dict(get_nested(config, "data.exp226_oof"))
    path = resolve_artifact(
        str(spec["filename"]),
        candidates=[str(value) for value in spec.get("candidates", [])],
        patterns=[str(value) for value in spec.get("patterns", [])],
    )
    decompressed_sha = sha256_gzip_decompressed(path)
    if decompressed_sha != str(spec["expected_decompressed_sha256"]):
        raise ValueError("exp226 OOF decompressed SHA changed")
    safe_columns = [str(value) for value in spec["safe_columns"]]
    frame = pd.read_csv(path, usecols=safe_columns, dtype={"well_id": str})
    forbidden = set(str(value) for value in spec["forbidden_score_columns"])
    leaked = sorted(forbidden.intersection(frame.columns))
    if leaked:
        raise ValueError(f"exp226 target-free input exposed forbidden columns: {leaked}")
    frame["well_id"] = frame["well_id"].astype(str)
    for column in ("row_idx", "suffix_offset", "fold"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.int64)
    frame["tvt_geop"] = pd.to_numeric(frame["tvt_geop"], errors="raise").astype(
        np.float64
    )
    frame = frame.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(
        drop=True
    )
    if frame.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("exp226 safe OOF contains duplicate row identities")
    if not np.isfinite(frame["tvt_geop"]).all():
        raise ValueError("exp226 tvt_geop must be finite")
    if len(frame) != int(get_nested(config, "validation.expected_rows")):
        raise ValueError("exp226 row count changed")
    if frame["well_id"].nunique() != int(get_nested(config, "validation.expected_wells")):
        raise ValueError("exp226 well count changed")
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    if sorted(frame["fold"].unique().tolist()) != expected_folds:
        raise ValueError("exp226 fold set changed")
    if not frame.groupby("well_id")["fold"].nunique().eq(1).all():
        raise ValueError("each exp226 well must belong to one reporting fold")
    return frame, path, {
        "name": "exp226_group_safe_oof_safe_columns",
        "path": str(path),
        "raw_sha256": sha256_path(path),
        "decompressed_sha256": decompressed_sha,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "safe_columns": safe_columns,
    }


def load_exp226_truth(
    path: Path,
    config: Mapping[str, Any],
    ledger: TruthAccessLedger,
) -> pd.DataFrame:
    ledger.require_frozen()
    columns = [str(value) for value in get_nested(config, "data.exp226_oof.truth_columns")]
    frame = pd.read_csv(path, usecols=columns, dtype={"well_id": str})
    frame["well_id"] = frame["well_id"].astype(str)
    frame["row_idx"] = pd.to_numeric(frame["row_idx"], errors="raise").astype(np.int64)
    frame["tvt_true"] = pd.to_numeric(frame["tvt_true"], errors="raise").astype(
        np.float64
    )
    frame = frame.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(
        drop=True
    )
    if frame.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("exp226 truth contains duplicate row identities")
    if not np.isfinite(frame["tvt_true"]).all():
        raise ValueError("exp226 truth must be finite")
    ledger.register_truth(len(frame))
    return frame


def load_horizontal_without_truth(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path, usecols=lambda column: column in {"id", "MD", "GR", "TVT_input"}
    )
    if "TVT" in frame.columns:
        raise ValueError("target-free horizontal reader exposed TVT")
    required = {"MD", "GR", "TVT_input"}
    if not required.issubset(frame.columns):
        raise ValueError(f"horizontal input missing {sorted(required - set(frame.columns))}")
    return frame


def load_hidden_roles(
    path: Path, config: Mapping[str, Any], ledger: TruthAccessLedger
) -> pd.DataFrame:
    ledger.require_frozen()
    expected = str(get_nested(config, "data.hidden_like.expected_sha256"))
    if sha256_path(path) != expected:
        raise ValueError("hidden-like assignment SHA changed")
    frame = pd.read_csv(path, dtype={"well_id": str})
    role_columns = [
        str(value)
        for value in dict(get_nested(config, "data.hidden_like.role_columns")).values()
    ]
    required = {"well_id", *role_columns}
    if not required.issubset(frame.columns):
        raise ValueError(f"hidden-like assignments missing {sorted(required - set(frame.columns))}")
    if frame["well_id"].duplicated().any():
        raise ValueError("hidden-like assignments require one row per well")
    ledger.register_hidden_roles(len(frame))
    return frame


def load_exp280_saved_control(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    spec = dict(get_nested(config, "data.exp280_saved_control"))
    score_path = resolve_artifact(
        str(spec["score_filename"]),
        patterns=[str(value) for value in spec.get("score_patterns", [])],
    )
    decompressed_sha = sha256_gzip_decompressed(score_path)
    if decompressed_sha != str(spec["expected_decompressed_sha256"]):
        raise ValueError("exp280 score decompressed SHA changed")
    contract_path = resolve_artifact(
        str(spec["contract_filename"]),
        patterns=[str(value) for value in spec.get("contract_patterns", [])],
    )
    contract = json.loads(contract_path.read_text())
    if bool(contract.get("truth_attached")):
        raise ValueError("exp280 saved control must be target-free")
    if contract.get("target_free_score_content_sha256") != str(
        spec["expected_content_sha256"]
    ):
        raise ValueError("exp280 declared score content SHA changed")
    if contract.get("scientific_contract_sha256") != str(
        spec["expected_scientific_contract_sha256"]
    ):
        raise ValueError("exp280 scientific contract SHA changed")
    scores = pd.read_csv(score_path, dtype={"well_id": str})
    forbidden = {"TVT", "tvt_true", "tvt_pred", "error", "abs_error", "gr_delta"}
    leaked = sorted(forbidden.intersection(scores.columns))
    if leaked:
        raise ValueError(f"exp280 control exposed truth/error columns: {leaked}")
    required = {
        "well_id",
        "fold",
        "block_id",
        "block_start_suffix_offset",
        "block_end_suffix_offset",
        "shift_slot",
        "shift_ft",
        "likelihood_mean",
        "likelihood_rank",
    }
    if not required.issubset(scores.columns):
        raise ValueError(f"exp280 control missing {sorted(required - set(scores.columns))}")
    for column in (
        "fold",
        "block_id",
        "block_start_suffix_offset",
        "block_end_suffix_offset",
        "shift_slot",
        "likelihood_rank",
    ):
        scores[column] = pd.to_numeric(scores[column], errors="raise").astype(np.int64)
    for column in ("shift_ft", "likelihood_mean"):
        scores[column] = pd.to_numeric(scores[column], errors="raise").astype(np.float64)
    scores["well_id"] = scores["well_id"].astype(str)
    scores = scores.sort_values(
        ["well_id", "block_id", "shift_slot"], kind="mergesort"
    ).reset_index(drop=True)
    sizes = scores.groupby(["well_id", "block_id"], sort=False).size()
    expected_blocks = int(get_nested(config, "validation.expected_blocks"))
    if len(sizes) != expected_blocks or not sizes.eq(len(EXPECTED_SHIFTS)).all():
        raise ValueError("exp280 block/candidate count changed")
    shift_matrix = scores["shift_ft"].to_numpy().reshape(-1, len(EXPECTED_SHIFTS))
    if not np.array_equal(
        shift_matrix, np.broadcast_to(EXPECTED_SHIFTS, shift_matrix.shape)
    ):
        raise ValueError("exp280 shift order changed")
    score_matrix = scores["likelihood_mean"].to_numpy().reshape(
        -1, len(EXPECTED_SHIFTS)
    )
    rank_matrix = scores["likelihood_rank"].to_numpy().reshape(
        -1, len(EXPECTED_SHIFTS)
    )
    recomputed = np.vstack([rank_descending(row) for row in score_matrix])
    if not np.array_equal(recomputed, rank_matrix):
        raise ValueError("exp280 saved ranks disagree with saved scores")
    actual_content_sha = dataframe_content_sha(scores)
    return scores, [
        {
            "name": "exp280_saved_target_free_scores",
            "path": str(score_path),
            "raw_sha256": sha256_path(score_path),
            "decompressed_sha256": decompressed_sha,
            "loaded_frame_content_sha256": actual_content_sha,
            "declared_content_sha256": str(spec["expected_content_sha256"]),
            "rows": len(scores),
        },
        {
            "name": "exp280_saved_scientific_contract",
            "path": str(contract_path),
            "raw_sha256": sha256_path(contract_path),
            "scientific_contract_sha256": contract["scientific_contract_sha256"],
        },
    ]


# %% [markdown]
# ## 5. Known-prefix affine posterior and fold-safe AR(1) prior
#
# The affine posterior is fit once per current well from finite known-prefix
# pairs. Candidate blocks never refit it. Per-well AR(1) estimates use only
# contiguous finite prefix residual pairs, and each reporting fold receives the
# Fisher-z median of outer-train wells only.

# %%
def prepare_typewell(typewell: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    required = {"TVT", "GR"}
    if not required.issubset(typewell.columns):
        raise ValueError(f"typewell missing {sorted(required - set(typewell.columns))}")
    frame = typewell[["TVT", "GR"]].copy()
    frame["TVT"] = pd.to_numeric(frame["TVT"], errors="coerce")
    frame["GR"] = pd.to_numeric(frame["GR"], errors="coerce")
    frame = frame.dropna(subset=["TVT"]).sort_values("TVT", kind="mergesort")
    frame["GR"] = frame["GR"].ffill().bfill()
    values = frame[["TVT", "GR"]].to_numpy(np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("typewell requires at least two finite TVT/GR rows")
    return values[:, 0], values[:, 1]


def fit_prefix_affine_posterior(
    horizontal_without_truth: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    well_id: str,
    fold: int,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    if "TVT" in horizontal_without_truth.columns:
        raise ValueError("prefix affine fit forbids horizontal TVT")
    typewell_tvt, typewell_gr = prepare_typewell(typewell)
    tvt_input = pd.to_numeric(
        horizontal_without_truth["TVT_input"], errors="coerce"
    ).to_numpy(np.float64)
    observed_gr = pd.to_numeric(
        horizontal_without_truth["GR"], errors="coerce"
    ).to_numpy(np.float64)
    prefix_positions = np.flatnonzero(np.isfinite(tvt_input))
    finite_positions = prefix_positions[np.isfinite(observed_gr[prefix_positions])]
    x = np.interp(tvt_input[finite_positions], typewell_tvt, typewell_gr)
    y = observed_gr[finite_positions]
    finite = np.isfinite(x) & np.isfinite(y)
    finite_positions = finite_positions[finite]
    x = x[finite]
    y = y[finite]

    sigma_low, sigma_high = [
        float(value) for value in get_nested(config, "model.sigma.clip")
    ]
    identity_residual = y - x
    sigma_unclipped = float(np.std(identity_residual)) if len(identity_residual) else float("nan")
    sigma = float(np.clip(sigma_unclipped, sigma_low, sigma_high))
    if not np.isfinite(sigma):
        sigma = sigma_low

    prior_mean = np.asarray(get_nested(config, "model.affine.prior_mean"), dtype=np.float64)
    prior_std = np.asarray(get_nested(config, "model.affine.prior_std"), dtype=np.float64)
    prior_cov = np.diag(prior_std**2)
    prior_precision = np.diag(1.0 / (prior_std**2))
    design = np.column_stack([np.ones(len(x), dtype=np.float64), x])
    posterior_mean = prior_mean.copy()
    posterior_cov = prior_cov.copy()
    if len(x):
        precision = prior_precision + design.T @ design / (sigma**2)
        posterior_cov = np.linalg.inv(precision)
        posterior_mean = posterior_cov @ (
            prior_precision @ prior_mean + design.T @ y / (sigma**2)
        )

    minimum_pairs = int(get_nested(config, "model.affine.minimum_prefix_pairs"))
    minimum_x_std = float(get_nested(config, "model.affine.minimum_typewell_gr_std"))
    x_std = float(np.std(x)) if len(x) else float("nan")
    finite_posterior = bool(
        np.isfinite(posterior_mean).all() and np.isfinite(posterior_cov).all()
    )
    positive_slope = bool(posterior_mean[1] > 0.0)
    eligible = bool(
        len(x) >= minimum_pairs
        and np.isfinite(x_std)
        and x_std >= minimum_x_std
        and finite_posterior
        and positive_slope
    )
    reasons = []
    if len(x) < minimum_pairs:
        reasons.append("prefix_pairs")
    if not np.isfinite(x_std) or x_std < minimum_x_std:
        reasons.append("typewell_gr_std")
    if not finite_posterior:
        reasons.append("posterior_nonfinite")
    if not positive_slope:
        reasons.append("posterior_nonpositive_slope")

    residual = y - design @ posterior_mean
    residual_frame = pd.DataFrame(
        {
            "well_id": str(well_id),
            "fold": int(fold),
            "row_idx": finite_positions.astype(np.int64),
            "residual": residual.astype(np.float64),
        }
    )
    posterior = {
        "well_id": str(well_id),
        "fold": int(fold),
        "known_prefix_rows": int(len(prefix_positions)),
        "finite_prefix_pairs": int(len(x)),
        "typewell_gr_std": x_std,
        "sigma_unclipped": sigma_unclipped,
        "sigma": sigma,
        "posterior_intercept": float(posterior_mean[0]),
        "posterior_slope": float(posterior_mean[1]),
        "posterior_cov_00": float(posterior_cov[0, 0]),
        "posterior_cov_01": float(posterior_cov[0, 1]),
        "posterior_cov_11": float(posterior_cov[1, 1]),
        "affine_eligible": eligible,
        "ineligible_reason": "|".join(reasons),
        "prefix_residual_content_sha256": dataframe_content_sha(residual_frame),
    }
    return posterior, residual_frame


def posterior_arrays(row: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(
        [row["posterior_intercept"], row["posterior_slope"]], dtype=np.float64
    )
    covariance = np.asarray(
        [
            [row["posterior_cov_00"], row["posterior_cov_01"]],
            [row["posterior_cov_01"], row["posterior_cov_11"]],
        ],
        dtype=np.float64,
    )
    return mean, covariance


def estimate_well_ar1(
    residual_frame: pd.DataFrame, *, minimum_pairs: int, rho_clip: Sequence[float]
) -> dict[str, Any]:
    frame = residual_frame.sort_values("row_idx", kind="mergesort")
    row_idx = frame["row_idx"].to_numpy(np.int64)
    residual = frame["residual"].to_numpy(np.float64)
    consecutive = np.diff(row_idx) == 1
    left = residual[:-1][consecutive]
    right = residual[1:][consecutive]
    denominator = float(np.dot(left, left)) if len(left) else 0.0
    evaluable = bool(
        len(left) >= int(minimum_pairs)
        and denominator > 0.0
        and np.isfinite(left).all()
        and np.isfinite(right).all()
    )
    rho_raw = float(np.dot(left, right) / denominator) if evaluable else float("nan")
    lower, upper = [float(value) for value in rho_clip]
    rho_clipped = float(np.clip(rho_raw, lower, upper)) if evaluable else float("nan")
    return {
        "lag1_pair_count": int(len(left)),
        "ar1_evaluable": evaluable,
        "rho_raw": rho_raw,
        "rho_clipped": rho_clipped,
        "contiguous_run_count": int(np.sum(np.r_[True, np.diff(row_idx) != 1]))
        if len(row_idx)
        else 0,
    }


def build_fold_ar1_priors(
    posterior_frame: pd.DataFrame,
    residuals_by_well: Mapping[str, pd.DataFrame],
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    minimum_pairs = int(get_nested(config, "model.ar1.minimum_lag1_pairs_per_source_well"))
    rho_clip = [float(value) for value in get_nested(config, "model.ar1.rho_clip")]
    per_well_rows = []
    for row in posterior_frame.to_dict(orient="records"):
        estimate = estimate_well_ar1(
            residuals_by_well[str(row["well_id"])],
            minimum_pairs=minimum_pairs,
            rho_clip=rho_clip,
        )
        per_well_rows.append({**row, **estimate})
    per_well = pd.DataFrame(per_well_rows).sort_values(
        "well_id", kind="mergesort"
    ).reset_index(drop=True)

    prior_rows = []
    folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    for fold in folds:
        outer_valid = set(per_well.loc[per_well["fold"].eq(fold), "well_id"].astype(str))
        source = per_well.loc[
            per_well["fold"].ne(fold)
            & per_well["affine_eligible"].astype(bool)
            & per_well["ar1_evaluable"].astype(bool)
        ].copy()
        if source.empty:
            raise ValueError(f"fold {fold} has no evaluable outer-train AR1 sources")
        source_wells = sorted(source["well_id"].astype(str).tolist())
        overlap = sorted(set(source_wells).intersection(outer_valid))
        fisher_z = np.arctanh(source["rho_clipped"].to_numpy(np.float64))
        rho_fold = float(np.tanh(np.median(fisher_z)))
        prior_rows.append(
            {
                "fold": fold,
                "rho_fold": rho_fold,
                "source_wells": len(source_wells),
                "source_lag1_pairs": int(source["lag1_pair_count"].sum()),
                "source_well_ids_sha256": mapping_sha256({"well_ids": source_wells}),
                "outer_valid_source_overlap": len(overlap),
            }
        )
    priors = pd.DataFrame(prior_rows).sort_values("fold", kind="mergesort")
    return per_well, priors.reset_index(drop=True)


# %% [markdown]
# ## 6. Raw-finite block predictive likelihood and 2×2 factorial
#
# For each contiguous finite run, the stationary AR(1) whitening transform is
# applied to both residual and design. The affine variants integrate the fixed
# prefix posterior with the matrix determinant lemma and rank-2 Woodbury
# identity. Identity variants set `theta=[0, 1]` and `V=0`.

# %%
def contiguous_run_indices(row_idx: Sequence[int]) -> list[np.ndarray]:
    positions = np.asarray(row_idx, dtype=np.int64)
    if not len(positions):
        return []
    starts = np.r_[0, np.flatnonzero(np.diff(positions) != 1) + 1]
    ends = np.r_[starts[1:], len(positions)]
    return [
        np.arange(start, end, dtype=np.int64)
        for start, end in zip(starts, ends, strict=True)
    ]


def whiten_ar1(
    residual: np.ndarray, design: np.ndarray, rho: float
) -> tuple[np.ndarray, np.ndarray]:
    residual = np.asarray(residual, dtype=np.float64)
    design = np.asarray(design, dtype=np.float64)
    if residual.ndim != 1 or design.shape != (len(residual), 2) or not len(residual):
        raise ValueError("AR1 whitening requires a non-empty n-vector and n×2 design")
    scale = math.sqrt(max(1.0 - float(rho) ** 2, 0.0))
    white_residual = np.empty_like(residual)
    white_design = np.empty_like(design)
    white_residual[0] = scale * residual[0]
    white_design[0] = scale * design[0]
    if len(residual) > 1:
        white_residual[1:] = residual[1:] - float(rho) * residual[:-1]
        white_design[1:] = design[1:] - float(rho) * design[:-1]
    return white_residual, white_design


def gaussian_predictive_logpdf_woodbury(
    observed: Sequence[float],
    expected_typewell_gr: Sequence[float],
    mean: Sequence[float],
    covariance: np.ndarray,
    *,
    sigma: float,
    rho: float,
) -> float:
    y = np.asarray(observed, dtype=np.float64)
    x = np.asarray(expected_typewell_gr, dtype=np.float64)
    posterior_mean = np.asarray(mean, dtype=np.float64)
    posterior_cov = np.asarray(covariance, dtype=np.float64)
    if len(y) != len(x) or not len(y):
        raise ValueError("predictive score requires aligned non-empty observations")
    design = np.column_stack([np.ones(len(x), dtype=np.float64), x])
    residual = y - design @ posterior_mean
    z, z_design = whiten_ar1(residual, design, float(rho))
    sigma2 = float(sigma) ** 2
    if sigma2 <= 0.0 or not np.isfinite(sigma2):
        raise ValueError("predictive sigma must be positive and finite")
    if np.allclose(posterior_cov, 0.0, atol=0.0, rtol=0.0):
        logdet = len(z) * math.log(sigma2)
        quadratic = float(np.dot(z, z) / sigma2)
    else:
        precision = np.linalg.inv(posterior_cov)
        middle = precision + z_design.T @ z_design / sigma2
        sign_v, logdet_v = np.linalg.slogdet(posterior_cov)
        sign_m, logdet_m = np.linalg.slogdet(middle)
        if sign_v <= 0 or sign_m <= 0:
            raise ValueError("affine posterior predictive covariance is not positive definite")
        logdet = len(z) * math.log(sigma2) + logdet_v + logdet_m
        projected = z_design.T @ z / sigma2
        quadratic = float(
            np.dot(z, z) / sigma2 - projected @ np.linalg.solve(middle, projected)
        )
    value = -0.5 * (len(z) * math.log(2.0 * math.pi) + logdet + quadratic)
    if not np.isfinite(value):
        raise ValueError("Woodbury predictive log density is non-finite")
    return float(value)


def gaussian_predictive_logpdf_dense(
    observed: Sequence[float],
    expected_typewell_gr: Sequence[float],
    mean: Sequence[float],
    covariance: np.ndarray,
    *,
    sigma: float,
    rho: float,
) -> float:
    y = np.asarray(observed, dtype=np.float64)
    x = np.asarray(expected_typewell_gr, dtype=np.float64)
    posterior_mean = np.asarray(mean, dtype=np.float64)
    posterior_cov = np.asarray(covariance, dtype=np.float64)
    design = np.column_stack([np.ones(len(x), dtype=np.float64), x])
    residual = y - design @ posterior_mean
    z, z_design = whiten_ar1(residual, design, float(rho))
    covariance_dense = float(sigma) ** 2 * np.eye(len(z)) + (
        z_design @ posterior_cov @ z_design.T
    )
    sign, logdet = np.linalg.slogdet(covariance_dense)
    if sign <= 0:
        raise ValueError("dense predictive covariance is not positive definite")
    quadratic = float(z @ np.linalg.solve(covariance_dense, z))
    return float(
        -0.5 * (len(z) * math.log(2.0 * math.pi) + logdet + quadratic)
    )


def dense_woodbury_parity() -> float:
    rng = np.random.default_rng(427)
    errors = []
    for length, rho in ((7, 0.0), (11, 0.45), (17, -0.30)):
        x = rng.normal(75.0, 12.0, size=length)
        y = 4.0 + 1.08 * x + rng.normal(0.0, 3.0, size=length)
        mean = np.asarray([1.5, 0.97])
        covariance = np.asarray([[3.0, -0.01], [-0.01, 0.004]])
        woodbury = gaussian_predictive_logpdf_woodbury(
            y, x, mean, covariance, sigma=12.0, rho=rho
        )
        dense = gaussian_predictive_logpdf_dense(
            y, x, mean, covariance, sigma=12.0, rho=rho
        )
        errors.append(abs(woodbury - dense))
    return float(max(errors))


def score_candidate_runs(
    observed: np.ndarray,
    expected: np.ndarray,
    row_idx: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    sigma: float,
    rho: float,
) -> float:
    total = 0.0
    total_rows = 0
    for run in contiguous_run_indices(row_idx):
        total += gaussian_predictive_logpdf_woodbury(
            observed[run],
            expected[run],
            mean,
            covariance,
            sigma=sigma,
            rho=rho,
        )
        total_rows += len(run)
    if total_rows != len(observed) or not total_rows:
        raise ValueError("predictive run split lost finite rows")
    return float(total / total_rows)


def score_well_target_free(
    oof_safe: pd.DataFrame,
    horizontal_without_truth: pd.DataFrame,
    typewell: pd.DataFrame,
    posterior: Mapping[str, Any],
    *,
    rho_fold: float,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if "TVT" in horizontal_without_truth.columns:
        raise ValueError("target-free scoring exposed horizontal TVT")
    forbidden = set(
        str(value)
        for value in get_nested(config, "data.exp226_oof.forbidden_score_columns")
    )
    leaked = sorted(forbidden.intersection(oof_safe.columns))
    if leaked:
        raise ValueError(f"target-free OOF contains forbidden columns: {leaked}")
    required = {"well_id", "row_idx", "suffix_offset", "fold", "tvt_geop"}
    if not required.issubset(oof_safe.columns):
        raise ValueError(f"safe OOF missing {sorted(required - set(oof_safe.columns))}")
    oof = oof_safe.sort_values("row_idx", kind="mergesort").reset_index(drop=True)
    if oof.empty or oof["well_id"].nunique() != 1 or oof["fold"].nunique() != 1:
        raise ValueError("score_well_target_free requires one well and one fold")
    suffix_offset = oof["suffix_offset"].to_numpy(np.int64)
    if not np.array_equal(suffix_offset, np.arange(len(oof), dtype=np.int64)):
        raise ValueError("suffix_offset must be contiguous from zero")
    row_idx = oof["row_idx"].to_numpy(np.int64)
    if row_idx.min() < 0 or row_idx.max() >= len(horizontal_without_truth):
        raise ValueError("exp226 row_idx is outside the horizontal frame")
    if horizontal_without_truth.iloc[row_idx]["TVT_input"].notna().any():
        raise ValueError("exp226 OOF rows must align to unknown suffix rows")

    typewell_tvt, typewell_gr = prepare_typewell(typewell)
    raw_gr_all = pd.to_numeric(
        horizontal_without_truth["GR"], errors="coerce"
    ).to_numpy(np.float64)
    md_all = pd.to_numeric(
        horizontal_without_truth["MD"], errors="raise"
    ).to_numpy(np.float64)
    known_positions = np.flatnonzero(
        np.isfinite(
            pd.to_numeric(
                horizontal_without_truth["TVT_input"], errors="coerce"
            ).to_numpy(np.float64)
        )
    )
    if not len(known_positions):
        raise ValueError("well has no known TVT_input prefix")
    last_known = int(known_positions[-1])
    md_since = md_all[row_idx] - md_all[last_known]
    raw_gr = raw_gr_all[row_idx]
    block_rows = int(get_nested(config, "audit.block_rows"))
    block_id = suffix_offset // block_rows
    geop = oof["tvt_geop"].to_numpy(np.float64)
    affine_mean, affine_covariance = posterior_arrays(posterior)
    identity_mean = np.asarray([0.0, 1.0], dtype=np.float64)
    identity_covariance = np.zeros((2, 2), dtype=np.float64)
    sigma = float(posterior["sigma"])
    well_id = str(oof["well_id"].iloc[0])
    fold = int(oof["fold"].iloc[0])
    minimum_rows = int(get_nested(config, "audit.minimum_finite_gr_rows"))
    minimum_fraction = float(get_nested(config, "audit.minimum_finite_gr_fraction"))
    extension = float(get_nested(config, "audit.typewell_extension_ft"))
    seed = int(get_nested(config, "audit.negative_control.seed"))

    score_rows: list[dict[str, Any]] = []
    negative_rows: list[dict[str, Any]] = []
    eligibility_rows: list[dict[str, Any]] = []
    for block in np.unique(block_id):
        block_positions = np.flatnonzero(block_id == block)
        finite_local = np.isfinite(raw_gr[block_positions])
        finite_positions = block_positions[finite_local]
        finite_count = int(len(finite_positions))
        finite_fraction = float(finite_count / len(block_positions))
        eligible = bool(
            finite_count >= minimum_rows
            and finite_fraction >= minimum_fraction
            and bool(posterior["affine_eligible"])
        )
        reasons = []
        if finite_count < minimum_rows:
            reasons.append("finite_gr_rows")
        if finite_fraction < minimum_fraction:
            reasons.append("finite_gr_fraction")
        if not bool(posterior["affine_eligible"]):
            reasons.append("affine_ineligible")
        metadata = {
            "well_id": well_id,
            "fold": fold,
            "block_id": int(block),
            "block_start_suffix_offset": int(suffix_offset[block_positions[0]]),
            "block_end_suffix_offset": int(suffix_offset[block_positions[-1]]),
            "block_start_row_idx": int(row_idx[block_positions[0]]),
            "block_end_row_idx": int(row_idx[block_positions[-1]]),
            "block_row_count": int(len(block_positions)),
            "finite_gr_rows": finite_count,
            "finite_gr_fraction": finite_fraction,
            "md_since_min_ft": float(np.min(md_since[block_positions])),
            "md_since_max_ft": float(np.max(md_since[block_positions])),
            "md_since_mid_ft": float(np.mean(md_since[block_positions])),
            "affine_eligible": bool(posterior["affine_eligible"]),
            "eligible_block": eligible,
            "ineligible_reason": "|".join(reasons),
        }
        eligibility_rows.append(metadata)
        if not eligible:
            continue

        observed = raw_gr[finite_positions]
        observed_row_idx = row_idx[finite_positions]
        candidate_tvt = geop[finite_positions, None] + EXPECTED_SHIFTS[None, :]
        candidate_expected = np.empty_like(candidate_tvt)
        for slot in range(len(EXPECTED_SHIFTS)):
            candidate_expected[:, slot] = np.interp(
                candidate_tvt[:, slot], typewell_tvt, typewell_gr
            )
        extended = (candidate_tvt >= typewell_tvt.min() - extension) & (
            candidate_tvt <= typewell_tvt.max() + extension
        )
        variant_scores: dict[str, np.ndarray] = {}
        for variant in FACTORIAL_VARIANTS:
            use_affine = variant in {"affine_iid", "affine_ar1"}
            use_ar1 = variant in {"identity_ar1", "affine_ar1"}
            mean = affine_mean if use_affine else identity_mean
            covariance = affine_covariance if use_affine else identity_covariance
            rho = float(rho_fold) if use_ar1 else 0.0
            values = np.empty(len(EXPECTED_SHIFTS), dtype=np.float64)
            for slot in range(len(EXPECTED_SHIFTS)):
                values[slot] = score_candidate_runs(
                    observed,
                    candidate_expected[:, slot],
                    observed_row_idx,
                    mean,
                    covariance,
                    sigma=sigma,
                    rho=rho,
                )
            ranks = rank_descending(values)
            variant_scores[variant] = values
            for slot, shift in enumerate(EXPECTED_SHIFTS):
                score_rows.append(
                    {
                        **metadata,
                        "variant": variant,
                        "shift_slot": int(slot),
                        "shift_ft": float(shift),
                        "score": float(values[slot]),
                        "rank": int(ranks[slot]),
                        "extended_typewell_coverage": float(extended[:, slot].mean()),
                    }
                )

        primary = variant_scores[PRIMARY_VARIANT]
        permutation = stable_score_permutation(
            well_id, fold, int(block), len(primary), seed=seed
        )
        shuffled = primary[permutation]
        shuffled_ranks = rank_descending(shuffled)
        for slot, shift in enumerate(EXPECTED_SHIFTS):
            negative_rows.append(
                {
                    **metadata,
                    "variant": NEGATIVE_CONTROL,
                    "shift_slot": int(slot),
                    "shift_ft": float(shift),
                    "source_shift_slot": int(permutation[slot]),
                    "score": float(shuffled[slot]),
                    "rank": int(shuffled_ranks[slot]),
                }
            )

    eligibility = pd.DataFrame(eligibility_rows).sort_values(
        ["well_id", "block_id"], kind="mergesort"
    )
    score_columns = [
        *eligibility.columns,
        "variant",
        "shift_slot",
        "shift_ft",
        "score",
        "rank",
        "extended_typewell_coverage",
    ]
    negative_columns = [
        *eligibility.columns,
        "variant",
        "shift_slot",
        "shift_ft",
        "source_shift_slot",
        "score",
        "rank",
    ]
    scores = pd.DataFrame(score_rows, columns=score_columns).sort_values(
        ["well_id", "block_id", "variant", "shift_slot"], kind="mergesort"
    )
    negative = pd.DataFrame(negative_rows, columns=negative_columns).sort_values(
        ["well_id", "block_id", "shift_slot"], kind="mergesort"
    )
    manifest = {
        "well_id": well_id,
        "fold": fold,
        "horizontal_rows": len(horizontal_without_truth),
        "evaluation_rows": len(oof),
        "blocks": len(eligibility),
        "eligible_blocks": int(eligibility["eligible_block"].sum()),
        "finite_evaluation_gr_share": float(np.isfinite(raw_gr).mean()),
        "affine_eligible": bool(posterior["affine_eligible"]),
        "rho_fold": float(rho_fold),
        "sigma": sigma,
    }
    return (
        scores.reset_index(drop=True),
        negative.reset_index(drop=True),
        eligibility.reset_index(drop=True),
        manifest,
    )


# %% [markdown]
# ## 7. Saved exp280 alignment and target-free bundle freeze

# %%
def align_saved_control(
    target_scores: pd.DataFrame,
    negative_scores: pd.DataFrame,
    eligibility: pd.DataFrame,
    saved_control: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ledger_keys = eligibility[
        [
            "well_id",
            "fold",
            "block_id",
            "block_start_suffix_offset",
            "block_end_suffix_offset",
        ]
    ].sort_values(["well_id", "block_id"], kind="mergesort")
    control_keys = saved_control[
        [
            "well_id",
            "fold",
            "block_id",
            "block_start_suffix_offset",
            "block_end_suffix_offset",
        ]
    ].drop_duplicates().sort_values(["well_id", "block_id"], kind="mergesort")
    if not ledger_keys.reset_index(drop=True).equals(control_keys.reset_index(drop=True)):
        raise ValueError("exp427 block ledger differs from the fixed exp280 surface")

    eligible = eligibility.loc[eligibility["eligible_block"].astype(bool)]
    expected_keys = eligible[["well_id", "fold", "block_id"]].merge(
        pd.DataFrame(
            {
                "shift_slot": np.arange(len(EXPECTED_SHIFTS), dtype=np.int64),
                "shift_ft": EXPECTED_SHIFTS,
            }
        ),
        how="cross",
    )
    control = saved_control[
        [
            "well_id",
            "fold",
            "block_id",
            "shift_slot",
            "shift_ft",
            "likelihood_mean",
            "likelihood_rank",
        ]
    ].rename(
        columns={
            "likelihood_mean": "score",
            "likelihood_rank": "rank",
        }
    )
    aligned = expected_keys.merge(
        control,
        on=["well_id", "fold", "block_id", "shift_slot", "shift_ft"],
        how="left",
        validate="one_to_one",
    )
    if len(aligned) != len(expected_keys) or aligned[["score", "rank"]].isna().any().any():
        raise ValueError("eligible exp427 blocks do not align exactly to exp280")
    aligned["variant"] = SAVED_CONTROL
    aligned["rank"] = aligned["rank"].astype(np.int64)
    aligned = aligned[
        ["well_id", "fold", "block_id", "variant", "shift_slot", "shift_ft", "score", "rank"]
    ].sort_values(["well_id", "block_id", "shift_slot"], kind="mergesort")

    candidate_counts = (
        target_scores.groupby(["well_id", "block_id", "variant"]).size()
        if not target_scores.empty
        else pd.Series(dtype=np.int64)
    )
    expected_groups = len(eligible) * len(FACTORIAL_VARIANTS)
    if len(candidate_counts) != expected_groups:
        raise ValueError("eligible blocks do not contain all four factorial variants")
    for _, part in target_scores.groupby(
        ["well_id", "block_id", "variant"], sort=False
    ):
        ordered = part.sort_values("shift_slot", kind="mergesort")
        if not np.array_equal(
            ordered["shift_ft"].to_numpy(np.float64), EXPECTED_SHIFTS
        ):
            raise ValueError("factorial score shift identity/order changed")
    negative_counts = negative_scores.groupby(["well_id", "block_id"]).size()
    if len(negative_counts) != len(eligible) or not negative_counts.eq(
        len(EXPECTED_SHIFTS)
    ).all():
        raise ValueError("negative control does not contain 13 shifts per eligible block")
    finite_coverage = (
        float(np.isfinite(target_scores["score"].to_numpy(np.float64)).mean())
        if len(target_scores)
        else 0.0
    )
    technical = {
        "score_finite_coverage": finite_coverage,
        "row_identity_coverage": float(len(control_keys) / len(ledger_keys))
        if len(ledger_keys)
        else 0.0,
        "minimum_candidate_count_per_eligible_block_variant": int(candidate_counts.min())
        if len(candidate_counts)
        else 0,
        "maximum_candidate_count_per_eligible_block_variant": int(candidate_counts.max())
        if len(candidate_counts)
        else 0,
        "saved_control_aligned_rows": len(aligned),
        "saved_control_aligned_blocks": int(
            aligned[["well_id", "block_id"]].drop_duplicates().shape[0]
        ),
    }
    return aligned.reset_index(drop=True), technical


def freeze_target_free_bundle(
    *,
    scientific_contract_payload: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
    posterior: pd.DataFrame,
    fold_priors: pd.DataFrame,
    eligibility: pd.DataFrame,
    scores: pd.DataFrame,
    negative: pd.DataFrame,
    saved_control: pd.DataFrame,
) -> dict[str, Any]:
    content = {
        "scientific_contract_sha256": scientific_contract_payload[
            "scientific_contract_sha256"
        ],
        "input_manifest_content_sha256": mapping_sha256(
            logical_manifest_payload(input_manifest)
        ),
        "prefix_affine_posterior_content_sha256": dataframe_content_sha(posterior),
        "fold_ar1_prior_content_sha256": dataframe_content_sha(fold_priors),
        "eligibility_content_sha256": dataframe_content_sha(eligibility),
        "target_free_scores_content_sha256": dataframe_content_sha(scores),
        "negative_control_content_sha256": dataframe_content_sha(negative),
        "saved_control_alignment_content_sha256": dataframe_content_sha(saved_control),
    }
    return {
        **content,
        "target_free_bundle_sha256": mapping_sha256(content),
        "truth_attached": False,
    }


# %% [markdown]
# ## 8. Late truth attachment and rank readouts

# %%
def _family_readout(
    scores: pd.DataFrame,
    candidate_rmse: np.ndarray,
    nearest_slot: int,
    *,
    prefix: str,
) -> dict[str, Any]:
    ordered = scores.sort_values("shift_slot", kind="mergesort")
    if len(ordered) != len(EXPECTED_SHIFTS):
        raise ValueError(f"{prefix} does not contain exactly 13 shifts")
    if not np.array_equal(ordered["shift_ft"].to_numpy(np.float64), EXPECTED_SHIFTS):
        raise ValueError(f"{prefix} shift identity/order changed")
    rank = int(ordered["rank"].iloc[nearest_slot])
    top1_slot = int(np.argmin(ordered["rank"].to_numpy(np.int64)))
    values = ordered["score"].to_numpy(np.float64)
    sorted_values = np.sort(values)[::-1]
    return {
        f"{prefix}_nearest_shift_rank": rank,
        f"{prefix}_top1_hit": bool(rank == 1),
        f"{prefix}_top3_hit": bool(rank <= 3),
        f"{prefix}_mrr": float(1.0 / rank),
        f"{prefix}_top1_shift_ft": float(EXPECTED_SHIFTS[top1_slot]),
        f"{prefix}_top1_regret_rmse": float(
            candidate_rmse[top1_slot] - candidate_rmse[nearest_slot]
        ),
        f"{prefix}_top1_margin": float(sorted_values[0] - sorted_values[1]),
    }


def build_block_readout(
    target_scores: pd.DataFrame,
    negative_scores: pd.DataFrame,
    saved_control: pd.DataFrame,
    eligibility: pd.DataFrame,
    oof_safe: pd.DataFrame,
    truth: pd.DataFrame,
    ledger: TruthAccessLedger,
) -> pd.DataFrame:
    ledger.require_frozen()
    merged = oof_safe.merge(
        truth, on=["well_id", "row_idx"], how="left", validate="one_to_one"
    )
    if len(merged) != len(oof_safe) or merged["tvt_true"].isna().any():
        raise ValueError("truth row identity attachment failed")
    rows = []
    eligible = eligibility.loc[eligibility["eligible_block"].astype(bool)]
    for meta in eligible.to_dict(orient="records"):
        well = str(meta["well_id"])
        block = int(meta["block_id"])
        block_rows = merged.loc[
            merged["well_id"].astype(str).eq(well)
            & merged["suffix_offset"].between(
                int(meta["block_start_suffix_offset"]),
                int(meta["block_end_suffix_offset"]),
            )
        ].sort_values("suffix_offset", kind="mergesort")
        if len(block_rows) != int(meta["block_row_count"]):
            raise ValueError(f"truth block row identity mismatch for {well}/{block}")
        geop = block_rows["tvt_geop"].to_numpy(np.float64)
        true_tvt = block_rows["tvt_true"].to_numpy(np.float64)
        errors = geop[:, None] + EXPECTED_SHIFTS[None, :] - true_tvt[:, None]
        candidate_rmse = np.sqrt(np.mean(errors**2, axis=0))
        nearest_slot = int(np.argmin(candidate_rmse))
        base = {
            **meta,
            "continuous_optimal_shift_ft": float(np.mean(true_tvt - geop)),
            "nearest_shift_slot": nearest_slot,
            "nearest_shift_ft": float(EXPECTED_SHIFTS[nearest_slot]),
            "base_rmse": float(np.sqrt(np.mean((geop - true_tvt) ** 2))),
            "nearest_shift_rmse": float(candidate_rmse[nearest_slot]),
        }
        for variant in FACTORIAL_VARIANTS:
            part = target_scores.loc[
                target_scores["well_id"].astype(str).eq(well)
                & target_scores["block_id"].eq(block)
                & target_scores["variant"].eq(variant)
            ]
            base.update(
                _family_readout(
                    part, candidate_rmse, nearest_slot, prefix=str(variant)
                )
            )
        negative = negative_scores.loc[
            negative_scores["well_id"].astype(str).eq(well)
            & negative_scores["block_id"].eq(block)
        ]
        base.update(
            _family_readout(
                negative, candidate_rmse, nearest_slot, prefix=NEGATIVE_CONTROL
            )
        )
        saved = saved_control.loc[
            saved_control["well_id"].astype(str).eq(well)
            & saved_control["block_id"].eq(block)
        ]
        base.update(
            _family_readout(saved, candidate_rmse, nearest_slot, prefix=SAVED_CONTROL)
        )
        rows.append(base)
    if not rows:
        raise ValueError("no eligible blocks survived target-free scoring")
    return pd.DataFrame(rows).sort_values(
        ["well_id", "block_id"], kind="mergesort"
    ).reset_index(drop=True)


# %% [markdown]
# ## 9. Scope metrics and technical/scientific AND gates

# %%
def metric_row(frame: pd.DataFrame, *, family: str, scope: str) -> dict[str, Any]:
    if frame.empty:
        raise ValueError(f"scope {scope} selected zero eligible blocks")
    return {
        "scope": scope,
        "family": family,
        "blocks": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "top1_rate": float(frame[f"{family}_top1_hit"].mean()),
        "top3_rate": float(frame[f"{family}_top3_hit"].mean()),
        "mrr": float(frame[f"{family}_mrr"].mean()),
        "mean_rank": float(frame[f"{family}_nearest_shift_rank"].mean()),
        "top1_regret_rmse_mean": float(frame[f"{family}_top1_regret_rmse"].mean()),
        "top1_regret_rmse_p90": float(
            frame[f"{family}_top1_regret_rmse"].quantile(0.90)
        ),
    }


def build_metric_tables(
    readout: pd.DataFrame,
    hidden_roles: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    families = (*FACTORIAL_VARIANTS, SAVED_CONTROL, NEGATIVE_CONTROL)
    factorial = pd.DataFrame(
        [metric_row(readout, family=family, scope="overall") for family in families]
    )
    fold_rows = []
    for fold, part in readout.groupby("fold", sort=True):
        for family in families:
            row = metric_row(part, family=family, scope=f"fold_{int(fold)}")
            row["fold"] = int(fold)
            fold_rows.append(row)
    scopes: list[tuple[str, pd.DataFrame]] = [("overall", readout)]
    long_limit = float(get_nested(config, "audit.scopes.long_tail_min_md_since_ft"))
    scopes.append(
        (
            "long_tail_1000_plus",
            readout.loc[readout["md_since_mid_ft"].ge(long_limit)],
        )
    )
    indexed = hidden_roles.set_index("well_id")
    for scope_name, role_column in dict(
        get_nested(config, "data.hidden_like.role_columns")
    ).items():
        valid_wells = set(
            indexed.index[indexed[str(role_column)].astype(str).eq("valid")].astype(str)
        )
        scopes.append(
            (
                str(scope_name),
                readout.loc[readout["well_id"].astype(str).isin(valid_wells)],
            )
        )
    scope_rows = []
    for scope, part in scopes:
        for family in families:
            scope_rows.append(metric_row(part, family=family, scope=scope))
    return (
        factorial.reset_index(drop=True),
        pd.DataFrame(fold_rows),
        pd.DataFrame(scope_rows),
    )


def _metric_lookup(
    table: pd.DataFrame, *, family: str, metric: str, scope: str | None = None
) -> float:
    selected = table.loc[table["family"].eq(family)]
    if scope is not None:
        selected = selected.loc[selected["scope"].eq(scope)]
    if len(selected) != 1:
        raise ValueError(f"metric lookup expected one row: {family}/{scope}/{metric}")
    return float(selected.iloc[0][metric])


def evaluate_gates(
    technical: Mapping[str, Any],
    factorial_metrics: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    scope_metrics: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    technical_gate = dict(get_nested(config, "validation.technical_gate"))
    scientific_gate = dict(get_nested(config, "validation.scientific_gate"))
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    technical_checks = {
        "score_finite_coverage": float(technical["score_finite_coverage"])
        >= float(technical_gate["required_score_finite_coverage"]),
        "row_identity_coverage": float(technical["row_identity_coverage"])
        >= float(technical_gate["required_row_identity_coverage"]),
        "candidate_count": int(
            technical["minimum_candidate_count_per_eligible_block_variant"]
        )
        == int(technical_gate["required_candidate_count_per_eligible_block"])
        and int(technical["maximum_candidate_count_per_eligible_block_variant"])
        == int(technical_gate["required_candidate_count_per_eligible_block"]),
        "eligible_well_fraction": float(technical["eligible_well_fraction"])
        >= float(technical_gate["minimum_eligible_well_fraction"]),
        "eligible_block_fraction": float(technical["eligible_block_fraction"])
        >= float(technical_gate["minimum_eligible_block_fraction"]),
        "affine_eligible_well_fraction": float(
            technical["affine_eligible_well_fraction"]
        )
        >= float(technical_gate["minimum_affine_eligible_well_fraction"]),
        "outer_valid_rho_source_overlap_zero": int(
            technical["maximum_outer_valid_rho_source_overlap"]
        )
        == 0,
        "fold_rho_strictly_inside_clip": float(technical["maximum_abs_fold_rho"])
        < float(technical_gate["require_abs_fold_rho_strictly_below"]),
        "dense_woodbury_parity": float(technical["dense_woodbury_max_abs_error"])
        <= float(technical_gate["maximum_dense_woodbury_score_abs_error"]),
        "truth_late": int(technical["truth_rows_before_freeze"]) == 0
        and int(technical["hidden_rows_before_freeze"]) == 0,
        "runtime": float(technical["runtime_seconds"])
        <= float(technical_gate["maximum_runtime_seconds"]),
        "peak_rss": float(technical["peak_rss_gb"])
        <= float(technical_gate["maximum_peak_rss_gb"]),
    }

    primary_mrr = _metric_lookup(
        factorial_metrics, family=PRIMARY_VARIANT, metric="mrr", scope="overall"
    )
    primary_top3 = _metric_lookup(
        factorial_metrics, family=PRIMARY_VARIANT, metric="top3_rate", scope="overall"
    )
    matched_mrr = _metric_lookup(
        factorial_metrics, family=MATCHED_CONTROL, metric="mrr", scope="overall"
    )
    matched_top3 = _metric_lookup(
        factorial_metrics, family=MATCHED_CONTROL, metric="top3_rate", scope="overall"
    )
    saved_mrr = _metric_lookup(
        factorial_metrics, family=SAVED_CONTROL, metric="mrr", scope="overall"
    )
    saved_top3 = _metric_lookup(
        factorial_metrics, family=SAVED_CONTROL, metric="top3_rate", scope="overall"
    )
    fold_index = fold_metrics.set_index(["fold", "family"])
    if sorted(fold_metrics["fold"].unique().tolist()) != expected_folds:
        raise ValueError("fold metrics do not contain the fixed five folds")

    def improved_folds(control: str, metric: str) -> int:
        return int(
            sum(
                float(fold_index.loc[(fold, PRIMARY_VARIANT), metric])
                > float(fold_index.loc[(fold, control), metric])
                for fold in expected_folds
            )
        )

    stress_scopes = (
        "long_tail_1000_plus",
        "hidden_like_spatial",
        "hidden_like_typewell_purged",
    )
    stress_checks = {}
    for scope in stress_scopes:
        for metric in ("mrr", "top3_rate"):
            primary_value = _metric_lookup(
                scope_metrics, family=PRIMARY_VARIANT, metric=metric, scope=scope
            )
            stress_checks[f"{scope}_{metric}_vs_both_controls"] = bool(
                primary_value
                >= _metric_lookup(
                    scope_metrics, family=MATCHED_CONTROL, metric=metric, scope=scope
                )
                and primary_value
                >= _metric_lookup(
                    scope_metrics, family=SAVED_CONTROL, metric=metric, scope=scope
                )
            )

    matched_mrr_folds = improved_folds(MATCHED_CONTROL, "mrr")
    matched_top3_folds = improved_folds(MATCHED_CONTROL, "top3_rate")
    saved_mrr_folds = improved_folds(SAVED_CONTROL, "mrr")
    saved_top3_folds = improved_folds(SAVED_CONTROL, "top3_rate")
    affine_only_folds = improved_folds("affine_iid", "mrr")
    ar1_only_folds = improved_folds("identity_ar1", "mrr")
    shuffle_mrr_folds = improved_folds(NEGATIVE_CONTROL, "mrr")
    shuffle_top3_folds = improved_folds(NEGATIVE_CONTROL, "top3_rate")
    min_control_folds = int(
        scientific_gate["minimum_improved_folds_vs_each_control_mrr"]
    )
    min_control_top3_folds = int(
        scientific_gate["minimum_improved_folds_vs_each_control_top3"]
    )
    min_single_folds = int(
        scientific_gate["minimum_improved_folds_vs_each_single_factor"]
    )
    required_shuffle_folds = int(scientific_gate["require_real_above_shuffled_folds"])
    scientific_checks = {
        "mrr_gain_vs_matched": primary_mrr - matched_mrr
        >= float(scientific_gate["minimum_mrr_gain_vs_matched_identity_iid"]),
        "top3_gain_vs_matched": primary_top3 - matched_top3
        >= float(scientific_gate["minimum_top3_gain_vs_matched_identity_iid"]),
        "mrr_gain_vs_saved": primary_mrr - saved_mrr
        >= float(scientific_gate["minimum_mrr_gain_vs_saved_exp280"]),
        "top3_gain_vs_saved": primary_top3 - saved_top3
        >= float(scientific_gate["minimum_top3_gain_vs_saved_exp280"]),
        "mrr_folds_vs_matched": matched_mrr_folds >= min_control_folds,
        "top3_folds_vs_matched": matched_top3_folds >= min_control_top3_folds,
        "mrr_folds_vs_saved": saved_mrr_folds >= min_control_folds,
        "top3_folds_vs_saved": saved_top3_folds >= min_control_top3_folds,
        "mrr_gain_vs_affine_iid": primary_mrr
        - _metric_lookup(
            factorial_metrics, family="affine_iid", metric="mrr", scope="overall"
        )
        >= float(scientific_gate["minimum_mrr_gain_vs_affine_iid"]),
        "mrr_gain_vs_identity_ar1": primary_mrr
        - _metric_lookup(
            factorial_metrics, family="identity_ar1", metric="mrr", scope="overall"
        )
        >= float(scientific_gate["minimum_mrr_gain_vs_identity_ar1"]),
        "mrr_folds_vs_affine_iid": affine_only_folds >= min_single_folds,
        "mrr_folds_vs_identity_ar1": ar1_only_folds >= min_single_folds,
        "real_above_shuffle_all_folds": shuffle_mrr_folds
        >= required_shuffle_folds
        and shuffle_top3_folds >= required_shuffle_folds,
        "top1_regret_p90_vs_saved": _metric_lookup(
            factorial_metrics,
            family=PRIMARY_VARIANT,
            metric="top1_regret_rmse_p90",
            scope="overall",
        )
        <= _metric_lookup(
            factorial_metrics,
            family=SAVED_CONTROL,
            metric="top1_regret_rmse_p90",
            scope="overall",
        ),
        **stress_checks,
    }
    technical_passed = bool(all(technical_checks.values()))
    scientific_passed = bool(all(scientific_checks.values()))
    passed = bool(technical_passed and scientific_passed)
    return {
        "passed": passed,
        "technical_passed": technical_passed,
        "scientific_passed": scientific_passed,
        "technical_checks": technical_checks,
        "scientific_checks": scientific_checks,
        "observed": {
            "primary_mrr": primary_mrr,
            "primary_top3": primary_top3,
            "matched_mrr": matched_mrr,
            "matched_top3": matched_top3,
            "saved_mrr": saved_mrr,
            "saved_top3": saved_top3,
            "improved_folds": {
                "matched_mrr": matched_mrr_folds,
                "matched_top3": matched_top3_folds,
                "saved_mrr": saved_mrr_folds,
                "saved_top3": saved_top3_folds,
                "affine_iid_mrr": affine_only_folds,
                "identity_ar1_mrr": ar1_only_folds,
                "shuffle_mrr": shuffle_mrr_folds,
                "shuffle_top3": shuffle_top3_folds,
            },
        },
        "decision": (
            "stage_0_passed_decoder_requires_new_experiment_and_approval"
            if passed
            else "stage_0_failed_close_without_rescue"
        ),
    }


# %% [markdown]
# ## 10. Kaggle CPU orchestration and generated artifacts

# %%
def run_stage_0_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    if not KAGGLE_WORKING_ROOT.exists() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError(
            "Full exp427 Stage 0 must run on Kaggle. EXPERIMENT_ALLOW_LOCAL=1 "
            "is reserved for an explicitly approved local smoke run."
        )
    contract = validate_scientific_contract(config, require_run_approval=True)
    started = time.time()
    ledger = TruthAccessLedger()
    raw_dir = train_data_dir(config)
    raw_identity, raw_manifest = validate_raw_well_identity(config, raw_dir)
    safe_oof, exp226_path, exp226_manifest = load_exp226_safe(config)
    saved_control_all, saved_manifests = load_exp280_saved_control(config)
    if set(raw_identity["well_id"]) != set(safe_oof["well_id"].astype(str)):
        raise ValueError("raw train and exp226 well sets differ")

    hidden_spec = dict(get_nested(config, "data.hidden_like"))
    hidden_path = resolve_artifact(
        str(hidden_spec["filename"]),
        candidates=[str(value) for value in hidden_spec.get("candidates", [])],
        patterns=[str(value) for value in hidden_spec.get("patterns", [])],
    )
    hidden_sha = sha256_path(hidden_path)
    if hidden_sha != str(hidden_spec["expected_sha256"]):
        raise ValueError("hidden-like assignment SHA changed")

    posterior_rows: list[dict[str, Any]] = []
    residuals_by_well: dict[str, pd.DataFrame] = {}
    for index, well in enumerate(raw_identity["well_id"].astype(str), start=1):
        horizontal = load_horizontal_without_truth(
            raw_dir / f"{well}__horizontal_well.csv"
        )
        typewell = pd.read_csv(raw_dir / f"{well}__typewell.csv")
        fold = int(safe_oof.loc[safe_oof["well_id"].eq(well), "fold"].iloc[0])
        posterior, residuals = fit_prefix_affine_posterior(
            horizontal,
            typewell,
            well_id=well,
            fold=fold,
            config=config,
        )
        posterior_rows.append(posterior)
        residuals_by_well[well] = residuals
        if index % 50 == 0 or index == len(raw_identity):
            print(f"prefix affine posteriors wells={index}/{len(raw_identity)}")
    posterior_frame = pd.DataFrame(posterior_rows).sort_values(
        "well_id", kind="mergesort"
    )
    posterior_frame, fold_priors = build_fold_ar1_priors(
        posterior_frame, residuals_by_well, config
    )
    rho_by_fold = fold_priors.set_index("fold")["rho_fold"].to_dict()

    score_parts: list[pd.DataFrame] = []
    negative_parts: list[pd.DataFrame] = []
    eligibility_parts: list[pd.DataFrame] = []
    well_manifests: list[dict[str, Any]] = []
    posterior_by_well = posterior_frame.set_index("well_id")
    for index, well in enumerate(raw_identity["well_id"].astype(str), start=1):
        horizontal_path = raw_dir / f"{well}__horizontal_well.csv"
        typewell_path = raw_dir / f"{well}__typewell.csv"
        horizontal = load_horizontal_without_truth(horizontal_path)
        typewell = pd.read_csv(typewell_path)
        posterior = posterior_by_well.loc[well].to_dict()
        posterior["well_id"] = well
        fold = int(posterior["fold"])
        scores, negative, eligibility, manifest = score_well_target_free(
            safe_oof.loc[safe_oof["well_id"].eq(well)],
            horizontal,
            typewell,
            posterior,
            rho_fold=float(rho_by_fold[fold]),
            config=config,
        )
        manifest.update(
            {
                "horizontal_path": str(horizontal_path),
                "horizontal_raw_sha256": sha256_path(horizontal_path),
                "typewell_path": str(typewell_path),
                "typewell_raw_sha256": sha256_path(typewell_path),
            }
        )
        score_parts.append(scores)
        negative_parts.append(negative)
        eligibility_parts.append(eligibility)
        well_manifests.append(manifest)
        if index % 25 == 0 or index == len(raw_identity):
            print(f"factorial block scoring wells={index}/{len(raw_identity)}")

    target_scores = pd.concat(score_parts, ignore_index=True).sort_values(
        ["well_id", "block_id", "variant", "shift_slot"], kind="mergesort"
    )
    negative_scores = pd.concat(negative_parts, ignore_index=True).sort_values(
        ["well_id", "block_id", "shift_slot"], kind="mergesort"
    )
    eligibility = pd.concat(eligibility_parts, ignore_index=True).sort_values(
        ["well_id", "block_id"], kind="mergesort"
    )
    if len(eligibility) != int(get_nested(config, "validation.expected_blocks")):
        raise ValueError("exp427 block ledger count changed")
    saved_control, alignment_technical = align_saved_control(
        target_scores, negative_scores, eligibility, saved_control_all
    )
    well_manifest = pd.DataFrame(well_manifests).sort_values(
        "well_id", kind="mergesort"
    )
    input_manifest = {
        "experiment": EXPERIMENT_NAME,
        "truth_attached": False,
        "raw_train": {
            **raw_manifest,
            "files": raw_identity.to_dict(orient="records"),
        },
        "exp226_oof": exp226_manifest,
        "exp280_saved_control": saved_manifests,
        "hidden_like_declared_only": {
            "path": str(hidden_path),
            "raw_sha256": hidden_sha,
            "roles_read": False,
        },
        "well_scoring": {
            "content_sha256": dataframe_content_sha(well_manifest),
            "records": well_manifest.to_dict(orient="records"),
        },
    }

    artifacts = artifact_dir()
    paths = {
        "scientific_contract": artifacts / f"{OUTPUT_PREFIX}_scientific_contract.json",
        "input_manifest": artifacts / f"{OUTPUT_PREFIX}_input_manifest.json",
        "prefix_affine_posterior": artifacts
        / f"{OUTPUT_PREFIX}_prefix_affine_posterior.csv.gz",
        "fold_ar1_prior": artifacts / f"{OUTPUT_PREFIX}_fold_ar1_prior.csv",
        "eligibility": artifacts / f"{OUTPUT_PREFIX}_eligibility.csv.gz",
        "target_free_scores": artifacts / f"{OUTPUT_PREFIX}_target_free_scores.csv.gz",
        "negative_control_scores": artifacts
        / f"{OUTPUT_PREFIX}_negative_control_scores.csv.gz",
        "block_readout": artifacts / f"{OUTPUT_PREFIX}_block_readout.csv.gz",
        "factorial_metrics": artifacts / f"{OUTPUT_PREFIX}_factorial_metrics.csv",
        "fold_metrics": artifacts / f"{OUTPUT_PREFIX}_fold_metrics.csv",
        "scope_metrics": artifacts / f"{OUTPUT_PREFIX}_scope_metrics.csv",
        "gate": artifacts / f"{OUTPUT_PREFIX}_gate.json",
        "summary": artifacts / f"{OUTPUT_PREFIX}_summary.json",
    }
    write_json(paths["scientific_contract"], contract)
    write_json(paths["input_manifest"], input_manifest)
    generated = {
        "prefix_affine_posterior": write_csv_gzip(
            posterior_frame, paths["prefix_affine_posterior"]
        ),
        "fold_ar1_prior": write_csv(fold_priors, paths["fold_ar1_prior"]),
        "eligibility": write_csv_gzip(eligibility, paths["eligibility"]),
        "target_free_scores": write_csv_gzip(
            target_scores, paths["target_free_scores"]
        ),
        "negative_control_scores": write_csv_gzip(
            negative_scores, paths["negative_control_scores"]
        ),
    }
    frozen = freeze_target_free_bundle(
        scientific_contract_payload=contract,
        input_manifest=input_manifest,
        posterior=posterior_frame,
        fold_priors=fold_priors,
        eligibility=eligibility,
        scores=target_scores,
        negative=negative_scores,
        saved_control=saved_control,
    )
    ledger.mark_frozen(frozen["target_free_bundle_sha256"])

    truth = load_exp226_truth(exp226_path, config, ledger)
    hidden_roles = load_hidden_roles(hidden_path, config, ledger)
    readout = build_block_readout(
        target_scores,
        negative_scores,
        saved_control,
        eligibility,
        safe_oof,
        truth,
        ledger,
    )
    factorial_metrics, fold_metrics, scope_metrics = build_metric_tables(
        readout, hidden_roles, config
    )
    eligible_blocks = eligibility.loc[eligibility["eligible_block"].astype(bool)]
    eligible_well_fraction = float(
        eligible_blocks["well_id"].nunique()
        / int(get_nested(config, "validation.expected_wells"))
    )
    affine_eligible_well_fraction = float(
        posterior_frame["affine_eligible"].astype(bool).mean()
    )
    runtime_seconds = time.time() - started
    technical = {
        **alignment_technical,
        "eligible_well_fraction": eligible_well_fraction,
        "eligible_block_fraction": float(
            eligibility["eligible_block"].astype(bool).mean()
        ),
        "affine_eligible_well_fraction": affine_eligible_well_fraction,
        "maximum_outer_valid_rho_source_overlap": int(
            fold_priors["outer_valid_source_overlap"].max()
        ),
        "maximum_abs_fold_rho": float(np.abs(fold_priors["rho_fold"]).max()),
        "dense_woodbury_max_abs_error": dense_woodbury_parity(),
        "truth_rows_before_freeze": ledger.truth_rows_before_freeze,
        "hidden_rows_before_freeze": ledger.hidden_rows_before_freeze,
        "runtime_seconds": runtime_seconds,
        "peak_rss_gb": peak_rss_gb(),
    }
    gate = evaluate_gates(
        technical, factorial_metrics, fold_metrics, scope_metrics, config
    )
    generated.update(
        {
            "block_readout": write_csv_gzip(readout, paths["block_readout"]),
            "factorial_metrics": write_csv(
                factorial_metrics, paths["factorial_metrics"]
            ),
            "fold_metrics": write_csv(fold_metrics, paths["fold_metrics"]),
            "scope_metrics": write_csv(scope_metrics, paths["scope_metrics"]),
        }
    )
    write_json(paths["gate"], gate)
    overall = factorial_metrics.set_index("family").to_dict(orient="index")
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": (
            "stage_0_completed_gate_passed"
            if gate["passed"]
            else "stage_0_completed_gate_failed"
        ),
        "route": "pf_beam",
        "stage": "stage_0",
        "runtime_seconds": runtime_seconds,
        "peak_rss_gb": technical["peak_rss_gb"],
        "rows": len(safe_oof),
        "wells": int(safe_oof["well_id"].nunique()),
        "blocks": len(eligibility),
        "eligible_blocks": int(eligibility["eligible_block"].sum()),
        "factorial_variants": list(FACTORIAL_VARIANTS),
        "primary": PRIMARY_VARIANT,
        "execution_counts": get_nested(config, "execution_contract.stage_0"),
        "parent_control_regenerated": False,
        "truth_attachment": {
            "stage": "after_all_target_free_scores_eligibility_priors_controls_and_manifest_freeze",
            **ledger.report(),
        },
        "frozen_bundle": frozen,
        "technical": technical,
        "overall": overall,
        "gate": gate,
        "generated_artifacts": generated,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "decision": gate["decision"],
        "hmm_run": False,
        "pf_run": False,
        "beam_run": False,
        "prediction_created": False,
        "inference_run": False,
        "submission_created": False,
    }
    write_json(paths["summary"], summary)
    write_json(
        metrics_output_path(),
        {
            "experiment": EXPERIMENT_NAME,
            "status": summary["status"],
            "route": "pf_beam",
            "stage": "stage_0",
            "cv": None,
            "public_lb": None,
            "private_lb": None,
            "metric": get_nested(config, "validation.metric"),
            "diagnostic": {
                "overall": overall,
                "gate": gate,
                "target_free_bundle_sha256": frozen["target_free_bundle_sha256"],
            },
            "execution_counts": get_nested(config, "execution_contract.stage_0"),
            "notes": (
                "Train-side truth-late rank audit only. No HMM/PF/Beam, model, "
                "prediction, inference, or submission was produced."
            ),
        },
    )
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 11. Setup and configuration preview

# %%
CONFIG = load_config()
CONTRACT = validate_scientific_contract(CONFIG, require_run_approval=False)
print("Experiment:", EXPERIMENT_NAME)
print("Route:", get_nested(CONFIG, "experiment.route"))
print("Parent:", get_nested(CONFIG, "lineage.parent"))
print("Status:", get_nested(CONFIG, "experiment.status"))
print("Factorial variants:", list(FACTORIAL_VARIANTS))
print("Primary:", PRIMARY_VARIANT)
print("Execution counts:", get_nested(CONFIG, "execution_contract.stage_0"))
print("Scientific contract SHA256:", CONTRACT["scientific_contract_sha256"])
print(
    "Kaggle Stage 0 approved:",
    bool(get_nested(CONFIG, "execution.kaggle_push_approved"))
    and bool(get_nested(CONFIG, "execution.run_stage_0")),
)


# %% [markdown]
# ## 12. Fail-closed Stage 0 entrypoint
#
# The pushed version-2 package explicitly enabled this one Stage 0 run. After
# completion, the checked-in config disables package/push/run approval and
# `train_run_on_push`; decoder, prediction, inference, and submission also remain
# fail-closed.

# %%
if in_notebook_runtime() and bool(get_nested(CONFIG, "execution.run_stage_0")):
    SUMMARY = run_stage_0_experiment(CONFIG)
elif in_notebook_runtime():
    print(
        "Stage 0 was not run. Compact self-contained implementation is ready, "
        "but Kaggle package/push/run approval remains false."
    )
