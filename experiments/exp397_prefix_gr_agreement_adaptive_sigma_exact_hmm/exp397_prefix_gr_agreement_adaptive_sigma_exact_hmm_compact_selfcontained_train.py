# %% [markdown]
# # exp397 prefix-GR agreement adaptive sigma exact HMM — Stage 0
#
# known prefix の raw finite horizontal GR と typewell GR の Pearson 相関だけで、
# exp209 `sigma_gr` の将来の well-level 係数 `1.0 / 1.3` を凍結する。
# Stage 0 は agreement surface の coverage、非退化、full-prefix / last-512 安定性だけを
# truth-free / 0-HMM で監査する。exact HMM、prediction、truth join、inference、
# submission はこの notebook には実装しない。

# %% [markdown]
# ## Contents
# 1. Imports and notebook-safe configuration
# 2. Runtime, SHA, and deterministic output helpers
# 3. Frozen scientific and execution contract
# 4. Target-free raw input and reporting-fold preflight
# 5. Full-prefix and last-512 GR agreement helpers
# 6. Coefficient freeze, stability readout, and Stage 0 gate
# 7. Stage 0 orchestration and generated artifacts
# 8. Setup and fixed contract checks
# 9. Fail-closed Stage 0 execution

# %% [markdown]
# ## 1. Imports and notebook-safe configuration

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    from IPython import get_ipython
    from IPython.display import display
except ImportError:  # pragma: no cover

    def get_ipython() -> None:
        return None

    def display(value: Any) -> None:
        print(value)


EXPERIMENT_NAME = "exp397_prefix_gr_agreement_adaptive_sigma_exact_hmm"
OUTPUT_PREFIX = EXPERIMENT_NAME
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
IMPORT_ONLY = os.environ.get("EXP397_IMPORT_ONLY", "0") == "1"
EXECUTE_NOTEBOOK = get_ipython() is not None and not IMPORT_ONLY
SAFE_HORIZONTAL_COLUMNS = ["GR", "TVT_input"]
FORBIDDEN_PREFREEZE_COLUMNS = {
    "TVT",
    "tvt_true",
    "error",
    "abs_error",
    "Formation",
    "formation",
}
COEFFICIENT_COLUMNS = [
    "well_id",
    "fold",
    "full_evaluable",
    "full_rho_gr",
    "full_pair_count",
    "full_fallback_reason",
    "sigma_multiplier",
    "coefficient_group",
]


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


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


def project_root() -> Path:
    for candidate in (PACKAGE_DIR, *PACKAGE_DIR.parents):
        if (candidate / "project.yml").is_file():
            return candidate
    return PACKAGE_DIR


def load_experiment_config() -> dict[str, Any]:
    root = project_root()
    candidates = (
        PACKAGE_DIR / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
        KAGGLE_WORKING_ROOT / "config.yaml",
    )
    for path in candidates:
        if not path.is_file():
            continue
        config = read_yaml(path)
        if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
            return config
    raise FileNotFoundError(f"Could not locate exp397 config; checked={candidates}")


# %% [markdown]
# ## 2. Runtime, SHA, and deterministic output helpers

# %%
def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        item = float(value)
        return item if math.isfinite(item) else None
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_gzip_csv(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    digest = hashlib.sha256()
    newline_count = 0
    last_byte = b""
    with gzip.open(source, "rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
            newline_count += chunk.count(b"\n")
            if chunk:
                last_byte = chunk[-1:]
    line_count = newline_count + int(bool(last_byte) and last_byte != b"\n")
    return {
        "path": str(source),
        "bytes": source.stat().st_size,
        "raw_sha256": sha256_path(source),
        "decompressed_sha256": digest.hexdigest(),
        "data_rows": max(0, line_count - 1),
    }


def mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def dataframe_content_sha256(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
) -> str:
    selected = list(frame.columns) if columns is None else list(columns)
    digest = hashlib.sha256()
    for column in selected:
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


def dataframe_schema_sha256(frame: pd.DataFrame) -> str:
    schema = [(str(column), str(dtype)) for column, dtype in frame.dtypes.items()]
    return hashlib.sha256(
        json.dumps(schema, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            to_jsonable(dict(payload)),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )
    return {"path": str(path), "raw_sha256": sha256_path(path)}


def write_frame(
    frame: pd.DataFrame,
    path: Path,
    *,
    sort_columns: Sequence[str],
) -> dict[str, Any]:
    ordered = frame.sort_values(list(sort_columns), kind="mergesort").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        ordered.to_csv(
            path,
            index=False,
            compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        )
    else:
        ordered.to_csv(path, index=False)
    manifest = {
        "path": str(path),
        "rows": int(len(ordered)),
        "columns": list(ordered.columns),
        "raw_sha256": sha256_path(path),
        "logical_content_sha256": dataframe_content_sha256(ordered),
        "schema_sha256": dataframe_schema_sha256(ordered),
    }
    if path.suffix == ".gz":
        manifest["decompressed_sha256"] = inspect_gzip_csv(path)[
            "decompressed_sha256"
        ]
    return manifest


def artifact_dir() -> Path:
    if KAGGLE_WORKING_ROOT.is_dir():
        output = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        output = project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    return output


def metrics_output_path() -> Path:
    if KAGGLE_WORKING_ROOT.is_dir():
        return KAGGLE_WORKING_ROOT / "metrics.json"
    return project_root() / "experiments" / EXPERIMENT_NAME / "metrics.json"


def runtime_is_kaggle() -> bool:
    return KAGGLE_INPUT_ROOT.is_dir() and KAGGLE_WORKING_ROOT.is_dir()


def runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyyaml": str(getattr(yaml, "__version__", "unknown")),
    }


# %% [markdown]
# ## 3. Frozen scientific and execution contract
#
# Stage 0 は diagnostic 1、reporting folds 5、HMM / model config / trained fold /
# PF / Beam / booster 各0。Stage 1 decoder はこの notebook に存在せず、Stage 0 PASS後も
# 別承認がなければ実装・実行しない。

# %%
def validate_scientific_contract(
    config: Mapping[str, Any],
    *,
    require_run_approval: bool = False,
) -> dict[str, Any]:
    expected = {
        "experiment.name": EXPERIMENT_NAME,
        "experiment.route": "pf_beam",
        "experiment.status": "stage_0_completed_guard_failed_closed",
        "lineage.parent": (
            "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation"
        ),
        "implementation.enabled": True,
        "implementation.scope": (
            "stage_0_compact_selfcontained_agreement_stability_audit"
        ),
        "implementation.stage_0_implemented": True,
        "implementation.stage_1_implemented": False,
        "implementation.canonical_notebook_adopted": True,
        "implementation.inference_enabled": False,
        "implementation.submission_enabled": False,
        "model.agreement.pair_policy": (
            "raw_finite_horizontal_gr_tvt_input_and_typewell_interp_only"
        ),
        "model.agreement.statistic": "pearson_correlation_float64",
        "model.agreement.minimum_primary_pairs": 64,
        "model.agreement.minimum_standard_deviation": 1.0e-6,
        "model.agreement.threshold": 0.50,
        "model.agreement.good_or_equal_multiplier": 1.0,
        "model.agreement.poor_multiplier": 1.3,
        "model.agreement.insufficient_support_fallback_multiplier": 1.0,
        "model.sigma.parent_clip": [10.0, 60.0],
        "model.sigma.application_order": "clip_parent_then_multiply_exactly_once",
        "model.sigma.post_multiplier_clip": None,
        "model.sigma.effective_range": [10.0, 78.0],
        "model.stage_0.truth_free": True,
        "model.stage_0.hmm_well_runs": 0,
        "model.stage_0.primary_window": "full_known_prefix",
        "model.stage_0.tail_window_raw_rows": 512,
        "model.stage_0.minimum_tail_pairs": 32,
        "model.stage_1.enabled_condition": (
            "stage_0_all_gates_pass_and_separate_user_approval"
        ),
        "execution_contract.parent_control_retraining": False,
        "execution.implementation_approved": True,
        "execution.run_stage_1": False,
        "execution.run_hmm": False,
        "execution.run_inference": False,
        "execution.create_submission": False,
        "runtime.kaggle.enable_gpu": False,
        "runtime.kaggle.enable_internet": False,
        "inference.enabled": False,
        "inference.create_submission": False,
    }
    for dotted_key, expected_value in expected.items():
        actual = get_nested(config, dotted_key)
        if actual != expected_value:
            raise ValueError(
                f"exp397 fixed contract mismatch: {dotted_key} must be "
                f"{expected_value!r}, got {actual!r}"
            )

    stage_0_expected = {
        "diagnostic_variants": 1,
        "reporting_folds": 5,
        "hmm_well_runs": 0,
        "model_configs": 0,
        "trained_folds": 0,
        "pf_well_runs": 0,
        "beam_well_runs": 0,
        "boosters": 0,
    }
    stage_0 = dict(get_nested(config, "execution_contract.stage_0", {}))
    if stage_0 != stage_0_expected:
        raise ValueError(f"exp397 Stage 0 execution contract changed: {stage_0}")

    gate_expected = {
        "minimum_primary_evaluable_well_fraction": 0.90,
        "maximum_fallback_well_fraction": 0.10,
        "poor_multiplier_well_fraction_range": [0.10, 0.90],
        "minimum_each_fold_primary_evaluable_fraction": 0.80,
        "minimum_tail_evaluable_well_fraction": 0.75,
        "minimum_full_tail_multiplier_agreement": 0.80,
        "minimum_full_tail_spearman_correlation": 0.70,
    }
    gate = dict(get_nested(config, "model.stage_0.pass_requires_all", {}))
    if gate != gate_expected:
        raise ValueError(f"exp397 Stage 0 gate contract changed: {gate}")

    forbidden = set(str(value) for value in get_nested(config, "model.forbidden", []))
    required_forbidden = {
        "continuous_or_quantile_multiplier_mapping",
        "threshold_support_window_or_multiplier_grid",
        "bias_rmse_ncc_dtw_acf_or_tail_primary_gate",
        "finite_only_or_mad_sigma",
        "row_varying_sigma",
        "huber_student_t_or_mixture_emission",
        "transition_prior_or_state_grid_change",
        "blend_weight_search",
        "parent_control_rerun",
        "same_oof_rescue",
    }
    if not required_forbidden.issubset(forbidden):
        raise ValueError("exp397 forbidden-operation contract is incomplete")

    if list(get_nested(config, "validation.expected_folds", [])) != list(range(5)):
        raise ValueError("exp397 reporting folds must remain [0, 1, 2, 3, 4]")

    if require_run_approval:
        approved = bool(get_nested(config, "execution.kaggle_push_approved"))
        train_approved = bool(get_nested(config, "execution.train_run_approved"))
        enabled = bool(get_nested(config, "execution.run_stage_0"))
        run_on_push = bool(get_nested(config, "runtime.kaggle.train_run_on_push"))
        if not (approved and train_approved and enabled and run_on_push):
            raise RuntimeError(
                "exp397 Stage 0 Kaggle package/push/run is not approved; "
                "all execution flags must be true"
            )
        if not runtime_is_kaggle():
            raise RuntimeError("The first full exp397 Stage 0 run must execute on Kaggle CPU")

    return stage_0


def build_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "parent": get_nested(config, "lineage.parent"),
        "stage": "stage_0_prefix_gr_agreement_stability",
        "target_free": True,
        "horizontal_columns_read": SAFE_HORIZONTAL_COLUMNS,
        "forbidden_prefreeze_columns": sorted(FORBIDDEN_PREFREEZE_COLUMNS),
        "agreement": get_nested(config, "model.agreement"),
        "sigma_contract": get_nested(config, "model.sigma"),
        "stage_0": get_nested(config, "model.stage_0"),
        "execution_contract": get_nested(config, "execution_contract.stage_0"),
        "parent_control_loaded": False,
        "hmm_implemented": False,
        "stage_1_implemented": False,
        "inference_enabled": False,
        "submission_enabled": False,
    }
    contract["scientific_contract_sha256"] = mapping_sha256(contract)
    return contract


# %% [markdown]
# ## 4. Target-free raw input and reporting-fold preflight
#
# horizontal reader は `GR` と `TVT_input` だけを明示的に読む。horizontal `TVT`、
# Formation、error、hidden-like role は agreement / coefficient freeze より前には読まない。
# exp226 OOF は SHA を検証した後、group-safe な `well_id / fold` だけを reporting strata として
# 使用する。exp209 prediction と saved LikPF は Stage 0 ではロードしない。

# %%
@dataclass
class TargetFreeLedger:
    forbidden_rows_before_freeze: int = 0
    frozen: bool = False

    def guard_horizontal_columns(
        self,
        columns: Sequence[str],
        rows: int,
        source: str,
    ) -> None:
        forbidden = sorted(set(columns) & FORBIDDEN_PREFREEZE_COLUMNS)
        if forbidden:
            self.forbidden_rows_before_freeze += int(rows)
            raise ValueError(
                f"forbidden pre-freeze horizontal columns from {source}: {forbidden}"
            )

    def freeze(self) -> None:
        if self.forbidden_rows_before_freeze != 0:
            raise RuntimeError("target-free coefficient freeze has prior truth access")
        self.frozen = True


def resolve_existing(filename: str, candidates: Sequence[str]) -> Path:
    root = project_root()
    checked: list[str] = []
    for raw in candidates:
        candidate = Path(str(raw))
        possible = (
            candidate,
            root / candidate,
            PACKAGE_DIR / candidate,
            candidate / filename,
            root / candidate / filename,
            PACKAGE_DIR / candidate / filename,
        )
        for path in possible:
            checked.append(str(path))
            if path.is_file() and path.stat().st_size > 0:
                return path
    for search_root in (KAGGLE_INPUT_ROOT, Path("/tmp")):
        if not search_root.exists():
            continue
        for path in sorted(search_root.rglob(filename)):
            checked.append(str(path))
            if path.is_file() and path.stat().st_size > 0:
                return path
    raise FileNotFoundError(f"Could not resolve {filename}; checked={checked[:80]}")


def resolve_raw_train_dir(config: Mapping[str, Any]) -> Path:
    candidates = [
        Path(str(value))
        for value in get_nested(config, "data.raw_train_candidates", [])
    ]
    candidates.append(
        project_root() / str(get_nested(config, "data.train_dir", "data/raw/train"))
    )
    for path in candidates:
        if path.is_dir() and next(path.glob("*__horizontal_well.csv"), None) is not None:
            return path
    if KAGGLE_INPUT_ROOT.exists():
        for path in sorted(KAGGLE_INPUT_ROOT.rglob("train")):
            if path.is_dir() and next(path.glob("*__horizontal_well.csv"), None):
                return path
    raise FileNotFoundError("Could not resolve the raw competition train directory")


def list_raw_wells(raw_dir: Path) -> list[str]:
    wells: list[str] = []
    for horizontal_path in sorted(raw_dir.glob("*__horizontal_well.csv")):
        well = horizontal_path.name.removesuffix("__horizontal_well.csv")
        if not (raw_dir / f"{well}__typewell.csv").is_file():
            raise FileNotFoundError(raw_dir / f"{well}__typewell.csv")
        wells.append(well)
    return wells


def validate_raw_well_identity(
    config: Mapping[str, Any],
    raw_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    for well in list_raw_wells(raw_dir):
        horizontal_path = raw_dir / f"{well}__horizontal_well.csv"
        typewell_path = raw_dir / f"{well}__typewell.csv"
        rows.append(
            {
                "well_id": well,
                "horizontal_raw_sha256": sha256_path(horizontal_path),
                "typewell_raw_sha256": sha256_path(typewell_path),
            }
        )
    manifest = pd.DataFrame(rows).sort_values("well_id", kind="mergesort").reset_index(
        drop=True
    )
    actual_sha = dataframe_content_sha256(
        manifest,
        ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"],
    )
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_sha = str(get_nested(config, "data.expected_raw_well_identity_sha256"))
    if len(manifest) != expected_wells:
        raise ValueError(f"raw well count mismatch: {len(manifest)} != {expected_wells}")
    if actual_sha != expected_sha:
        raise ValueError(
            f"raw well identity mismatch: expected={expected_sha} actual={actual_sha}"
        )
    return manifest, {
        "path": str(raw_dir),
        "wells": int(len(manifest)),
        "logical_content_sha256": actual_sha,
    }


def load_fold_assignment(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spec = dict(get_nested(config, "data.fold_assignment", {}))
    path = resolve_existing(
        str(spec["filename"]),
        [str(value) for value in spec.get("candidates", [])],
    )
    inspection = inspect_gzip_csv(path)
    expected_sha = str(spec["expected_decompressed_sha256"])
    if inspection["decompressed_sha256"] != expected_sha:
        raise ValueError(
            "exp226 fold input decompressed SHA mismatch: "
            f"expected={expected_sha} actual={inspection['decompressed_sha256']}"
        )
    frame = pd.read_csv(
        path,
        usecols=["well_id", "fold"],
        dtype={"well_id": str},
    )
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(np.int64)
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    if len(frame) != expected_rows:
        raise ValueError(f"fold input row mismatch: {len(frame)} != {expected_rows}")
    grouped = frame.groupby("well_id", sort=True)["fold"].agg(["nunique", "first"])
    if not bool(grouped["nunique"].eq(1).all()):
        raise ValueError("each well must have exactly one reporting fold")
    assignment = grouped["first"].rename("fold").reset_index()
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_folds = [
        int(value) for value in get_nested(config, "validation.expected_folds")
    ]
    if len(assignment) != expected_wells:
        raise ValueError(f"fold well count mismatch: {len(assignment)} != {expected_wells}")
    if sorted(assignment["fold"].unique().tolist()) != expected_folds:
        raise ValueError("reporting fold set differs from [0, 1, 2, 3, 4]")
    manifest = {
        "name": "exp226_group_safe_reporting_fold",
        **inspection,
        "rows": int(len(frame)),
        "wells": int(len(assignment)),
        "folds": expected_folds,
        "assignment_logical_content_sha256": dataframe_content_sha256(
            assignment,
            ["well_id", "fold"],
        ),
        "columns_read": ["well_id", "fold"],
    }
    return assignment, manifest


def load_target_free_well(
    raw_dir: Path,
    well: str,
    ledger: TargetFreeLedger,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    horizontal_path = raw_dir / f"{well}__horizontal_well.csv"
    typewell_path = raw_dir / f"{well}__typewell.csv"
    horizontal = pd.read_csv(
        horizontal_path,
        usecols=SAFE_HORIZONTAL_COLUMNS,
    )
    ledger.guard_horizontal_columns(
        list(horizontal.columns),
        len(horizontal),
        horizontal_path.name,
    )
    if list(horizontal.columns) != SAFE_HORIZONTAL_COLUMNS:
        raise ValueError(
            f"{well} target-free horizontal schema mismatch: {horizontal.columns.tolist()}"
        )
    typewell = pd.read_csv(typewell_path, usecols=["TVT", "GR"])
    return horizontal, typewell, horizontal_path, typewell_path


# %% [markdown]
# ## 5. Full-prefix and last-512 GR agreement helpers
#
# Pair は `TVT_input` が finite な known-prefix raw row を先に固定し、その row の raw `GR` と
# typewell interpolation がともに finite な場合だけ残す。last-512 は finite pair を512個
# 選ぶのではなく、known prefix の末尾512 raw rowsを先に固定する。typewell は TVT の
# stable sort後にGRをforward/backward fillし、`np.interp` のendpoint holdを使う。

# %%
def prepare_typewell(typewell: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    frame = typewell[["TVT", "GR"]].copy()
    frame["TVT"] = pd.to_numeric(frame["TVT"], errors="coerce")
    frame["GR"] = pd.to_numeric(frame["GR"], errors="coerce")
    frame = frame.dropna(subset=["TVT"]).sort_values("TVT", kind="mergesort")
    frame["GR"] = frame["GR"].ffill().bfill()
    values = frame[["TVT", "GR"]].to_numpy(np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("typewell requires at least two finite TVT/filled-GR rows")
    return values[:, 0], values[:, 1]


def pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if len(left_array) != len(right_array) or len(left_array) < 2:
        return float("nan")
    left_centered = left_array - float(np.mean(left_array))
    right_centered = right_array - float(np.mean(right_array))
    denominator = float(
        np.sqrt(
            np.dot(left_centered, left_centered)
            * np.dot(right_centered, right_centered)
        )
    )
    if not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    return float(np.dot(left_centered, right_centered) / denominator)


def select_sigma_multiplier(
    *,
    pair_count: int,
    horizontal_std: float,
    typewell_std: float,
    rho_gr: float,
    minimum_pairs: int,
    minimum_standard_deviation: float,
    threshold: float,
    poor_multiplier: float,
    fallback_multiplier: float,
) -> tuple[bool, float, str]:
    reasons = []
    if int(pair_count) < int(minimum_pairs):
        reasons.append("insufficient_pair_count")
    if not np.isfinite(horizontal_std) or horizontal_std <= minimum_standard_deviation:
        reasons.append("horizontal_std_at_or_below_minimum")
    if not np.isfinite(typewell_std) or typewell_std <= minimum_standard_deviation:
        reasons.append("typewell_std_at_or_below_minimum")
    if not np.isfinite(rho_gr):
        reasons.append("nonfinite_rho")
    evaluable = not reasons
    if not evaluable:
        return False, float(fallback_multiplier), "|".join(reasons)
    multiplier = poor_multiplier if rho_gr < threshold else fallback_multiplier
    return True, float(multiplier), ""


def estimate_agreement_window(
    horizontal_without_truth: pd.DataFrame,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    *,
    window: str,
    tail_rows: int,
    minimum_pairs: int,
    minimum_standard_deviation: float,
    threshold: float,
    poor_multiplier: float,
    fallback_multiplier: float,
) -> dict[str, Any]:
    if set(horizontal_without_truth.columns) != set(SAFE_HORIZONTAL_COLUMNS):
        raise ValueError("agreement input must contain only GR and TVT_input")
    raw_gr = pd.to_numeric(
        horizontal_without_truth["GR"],
        errors="coerce",
    ).to_numpy(np.float64)
    tvt_input = pd.to_numeric(
        horizontal_without_truth["TVT_input"],
        errors="coerce",
    ).to_numpy(np.float64)
    known_positions = np.flatnonzero(np.isfinite(tvt_input))
    if window == "full_known_prefix":
        window_positions = known_positions
    elif window == "last_512_known_prefix_raw_rows":
        window_positions = known_positions[-int(tail_rows) :]
    else:
        raise ValueError(f"unknown agreement window: {window}")

    if len(window_positions):
        reference = np.interp(
            tvt_input[window_positions],
            typewell_tvt,
            typewell_gr,
        )
        finite = (
            np.isfinite(tvt_input[window_positions])
            & np.isfinite(raw_gr[window_positions])
            & np.isfinite(reference)
        )
        pair_positions = window_positions[finite]
        horizontal_values = raw_gr[pair_positions]
        typewell_values = reference[finite]
    else:
        pair_positions = np.empty(0, dtype=np.int64)
        horizontal_values = np.empty(0, dtype=np.float64)
        typewell_values = np.empty(0, dtype=np.float64)

    pair_count = int(len(pair_positions))
    horizontal_std = (
        float(np.std(horizontal_values, ddof=0)) if pair_count else float("nan")
    )
    typewell_std = (
        float(np.std(typewell_values, ddof=0)) if pair_count else float("nan")
    )
    rho_gr = pearson_correlation(horizontal_values, typewell_values)
    mean_bias = (
        float(np.mean(horizontal_values - typewell_values))
        if pair_count
        else float("nan")
    )
    normalized_bias = (
        float(mean_bias / typewell_std)
        if np.isfinite(mean_bias)
        and np.isfinite(typewell_std)
        and typewell_std > minimum_standard_deviation
        else float("nan")
    )
    evaluable, multiplier, fallback_reason = select_sigma_multiplier(
        pair_count=pair_count,
        horizontal_std=horizontal_std,
        typewell_std=typewell_std,
        rho_gr=rho_gr,
        minimum_pairs=minimum_pairs,
        minimum_standard_deviation=minimum_standard_deviation,
        threshold=threshold,
        poor_multiplier=poor_multiplier,
        fallback_multiplier=fallback_multiplier,
    )
    return {
        "window": window,
        "known_raw_row_count": int(len(window_positions)),
        "pair_count": pair_count,
        "first_pair_row_idx": int(pair_positions[0]) if pair_count else -1,
        "last_pair_row_idx": int(pair_positions[-1]) if pair_count else -1,
        "horizontal_std": horizontal_std,
        "typewell_std": typewell_std,
        "mean_bias": mean_bias,
        "normalized_bias": normalized_bias,
        "rho_gr": rho_gr,
        "evaluable": bool(evaluable),
        "multiplier": multiplier,
        "fallback_reason": fallback_reason,
    }


def build_well_agreement(
    horizontal_without_truth: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    well_id: str,
    fold: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    agreement = dict(get_nested(config, "model.agreement", {}))
    stage_0 = dict(get_nested(config, "model.stage_0", {}))
    typewell_tvt, typewell_gr = prepare_typewell(typewell)
    shared = {
        "tail_rows": int(stage_0["tail_window_raw_rows"]),
        "minimum_standard_deviation": float(
            agreement["minimum_standard_deviation"]
        ),
        "threshold": float(agreement["threshold"]),
        "poor_multiplier": float(agreement["poor_multiplier"]),
        "fallback_multiplier": float(
            agreement["insufficient_support_fallback_multiplier"]
        ),
    }
    full = estimate_agreement_window(
        horizontal_without_truth,
        typewell_tvt,
        typewell_gr,
        window="full_known_prefix",
        minimum_pairs=int(agreement["minimum_primary_pairs"]),
        **shared,
    )
    tail = estimate_agreement_window(
        horizontal_without_truth,
        typewell_tvt,
        typewell_gr,
        window="last_512_known_prefix_raw_rows",
        minimum_pairs=int(stage_0["minimum_tail_pairs"]),
        **shared,
    )
    row: dict[str, Any] = {"well_id": str(well_id), "fold": int(fold)}
    for prefix, values in (("full", full), ("tail", tail)):
        for key, value in values.items():
            if key != "window":
                row[f"{prefix}_{key}"] = value
    row["sigma_multiplier"] = float(full["multiplier"])
    row["coefficient_group"] = (
        "fallback_parent_noop"
        if not bool(full["evaluable"])
        else (
            "poor_agreement_sigma_x1p3"
            if float(full["multiplier"]) == float(agreement["poor_multiplier"])
            else "good_agreement_parent_noop"
        )
    )
    return row


def apply_multiplier_to_clipped_parent_sigma(
    parent_sigma_clipped: float | np.ndarray,
    sigma_multiplier: float | np.ndarray,
) -> np.ndarray:
    base = np.asarray(parent_sigma_clipped, dtype=np.float64)
    multiplier = np.asarray(sigma_multiplier, dtype=np.float64)
    if not np.isfinite(base).all() or ((base < 10.0) | (base > 60.0)).any():
        raise ValueError("parent sigma must already be clipped to [10, 60]")
    if not np.isfinite(multiplier).all() or not np.isin(multiplier, [1.0, 1.3]).all():
        raise ValueError("exp397 multiplier must be exactly 1.0 or 1.3")
    return base * multiplier


# %% [markdown]
# ## 6. Coefficient freeze, stability readout, and Stage 0 gate
#
# full-prefix coefficient と agreement table は suffix truth を一度も読まずに row-order-sensitive
# logical SHAで凍結する。stability は full / tail の両方が evaluable なwellだけで計算し、
# fallback `1.0` 同士による見かけの一致でgateを通さない。

# %%
def freeze_target_free_agreement(
    agreement_schedule: pd.DataFrame,
    ledger: TargetFreeLedger,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    ordered = agreement_schedule.sort_values("well_id", kind="mergesort").reset_index(
        drop=True
    )
    if ordered["well_id"].duplicated().any():
        raise ValueError("agreement schedule must have one row per well")
    if list(ordered["well_id"]) != sorted(ordered["well_id"].astype(str).tolist()):
        raise ValueError("agreement schedule must be frozen in sorted well order")
    missing = [column for column in COEFFICIENT_COLUMNS if column not in ordered]
    if missing:
        raise ValueError(f"coefficient schedule missing columns: {missing}")
    ledger.freeze()
    coefficient = ordered[COEFFICIENT_COLUMNS].copy()
    freeze: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "stage": "stage_0_target_free_coefficient_freeze",
        "rows": int(len(ordered)),
        "wells": int(ordered["well_id"].nunique()),
        "agreement_schema_sha256": dataframe_schema_sha256(ordered),
        "agreement_logical_content_sha256": dataframe_content_sha256(ordered),
        "coefficient_schema_sha256": dataframe_schema_sha256(coefficient),
        "coefficient_logical_content_sha256": dataframe_content_sha256(coefficient),
        "truth_rows_accessed_before_freeze": int(
            ledger.forbidden_rows_before_freeze
        ),
        "horizontal_columns_read": SAFE_HORIZONTAL_COLUMNS,
        "threshold": float(get_nested(config, "model.agreement.threshold")),
        "minimum_primary_pairs": int(
            get_nested(config, "model.agreement.minimum_primary_pairs")
        ),
        "minimum_tail_pairs": int(
            get_nested(config, "model.stage_0.minimum_tail_pairs")
        ),
        "tail_window_raw_rows": int(
            get_nested(config, "model.stage_0.tail_window_raw_rows")
        ),
        "multipliers": [1.0, 1.3],
        "hmm_well_runs": 0,
        "parent_control_loaded": False,
        "prediction_generated": False,
    }
    freeze["freeze_manifest_sha256"] = mapping_sha256(freeze)
    return freeze


def verify_target_free_freeze(
    agreement_schedule: pd.DataFrame,
    freeze: Mapping[str, Any],
) -> None:
    ordered = agreement_schedule.sort_values("well_id", kind="mergesort").reset_index(
        drop=True
    )
    current_agreement_sha = dataframe_content_sha256(ordered)
    current_coefficient_sha = dataframe_content_sha256(
        ordered[COEFFICIENT_COLUMNS]
    )
    if current_agreement_sha != freeze["agreement_logical_content_sha256"]:
        raise RuntimeError("agreement schedule changed after target-free SHA freeze")
    if current_coefficient_sha != freeze["coefficient_logical_content_sha256"]:
        raise RuntimeError("coefficient table changed after target-free SHA freeze")
    if int(freeze["truth_rows_accessed_before_freeze"]) != 0:
        raise RuntimeError("target-free freeze recorded forbidden truth access")


def spearman_rank_correlation(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left_array) & np.isfinite(right_array)
    if int(finite.sum()) < 2:
        return float("nan")
    left_rank = pd.Series(left_array[finite]).rank(method="average").to_numpy(
        np.float64
    )
    right_rank = pd.Series(right_array[finite]).rank(method="average").to_numpy(
        np.float64
    )
    return pearson_correlation(left_rank, right_rank)


def finite_quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return {key: float("nan") for key in ("q00", "q05", "q25", "q50", "q75", "q95", "q100")}
    probabilities = [0.0, 0.05, 0.25, 0.50, 0.75, 0.95, 1.0]
    labels = ["q00", "q05", "q25", "q50", "q75", "q95", "q100"]
    return {
        label: float(value)
        for label, value in zip(labels, np.quantile(array, probabilities), strict=True)
    }


def build_stability_readout(
    agreement_schedule: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    stability = agreement_schedule.sort_values(
        "well_id",
        kind="mergesort",
    ).reset_index(drop=True).copy()
    stability["joint_evaluable"] = (
        stability["full_evaluable"].astype(bool)
        & stability["tail_evaluable"].astype(bool)
    )
    stability["full_tail_multiplier_match"] = np.where(
        stability["joint_evaluable"],
        stability["full_multiplier"].eq(stability["tail_multiplier"]),
        False,
    )
    stability["full_tail_rho_delta"] = (
        stability["full_rho_gr"].to_numpy(np.float64)
        - stability["tail_rho_gr"].to_numpy(np.float64)
    )

    fold_rows = []
    for fold, part in stability.groupby("fold", sort=True):
        joint = part.loc[part["joint_evaluable"]]
        fold_rows.append(
            {
                "fold": int(fold),
                "wells": int(len(part)),
                "full_evaluable_wells": int(part["full_evaluable"].sum()),
                "full_evaluable_fraction": float(part["full_evaluable"].mean()),
                "fallback_wells": int((~part["full_evaluable"].astype(bool)).sum()),
                "fallback_fraction": float((~part["full_evaluable"].astype(bool)).mean()),
                "poor_multiplier_wells": int(part["sigma_multiplier"].eq(1.3).sum()),
                "poor_multiplier_fraction": float(
                    part["sigma_multiplier"].eq(1.3).mean()
                ),
                "tail_evaluable_wells": int(part["tail_evaluable"].sum()),
                "tail_evaluable_fraction": float(part["tail_evaluable"].mean()),
                "joint_evaluable_wells": int(len(joint)),
                "full_tail_multiplier_agreement": (
                    float(joint["full_tail_multiplier_match"].mean())
                    if len(joint)
                    else float("nan")
                ),
                "full_tail_spearman_correlation": spearman_rank_correlation(
                    joint["full_rho_gr"],
                    joint["tail_rho_gr"],
                ),
            }
        )
    fold_metrics = pd.DataFrame(fold_rows)
    joint = stability.loc[stability["joint_evaluable"]]
    pooled = {
        "expected_wells": int(get_nested(config, "validation.expected_wells")),
        "actual_wells": int(len(stability)),
        "expected_folds": [
            int(value) for value in get_nested(config, "validation.expected_folds")
        ],
        "actual_folds": sorted(stability["fold"].astype(int).unique().tolist()),
        "full_evaluable_wells": int(stability["full_evaluable"].sum()),
        "full_evaluable_fraction": float(stability["full_evaluable"].mean()),
        "fallback_wells": int((~stability["full_evaluable"].astype(bool)).sum()),
        "fallback_fraction": float((~stability["full_evaluable"].astype(bool)).mean()),
        "poor_multiplier_wells": int(stability["sigma_multiplier"].eq(1.3).sum()),
        "poor_multiplier_fraction": float(
            stability["sigma_multiplier"].eq(1.3).mean()
        ),
        "tail_evaluable_wells": int(stability["tail_evaluable"].sum()),
        "tail_evaluable_fraction": float(stability["tail_evaluable"].mean()),
        "joint_evaluable_wells": int(len(joint)),
        "full_tail_multiplier_agreement": (
            float(joint["full_tail_multiplier_match"].mean())
            if len(joint)
            else float("nan")
        ),
        "full_tail_spearman_correlation": spearman_rank_correlation(
            joint["full_rho_gr"],
            joint["tail_rho_gr"],
        ),
        "minimum_fold_full_evaluable_fraction": (
            float(fold_metrics["full_evaluable_fraction"].min())
            if len(fold_metrics)
            else float("nan")
        ),
        "full_pair_count_quantiles": finite_quantiles(stability["full_pair_count"]),
        "tail_pair_count_quantiles": finite_quantiles(stability["tail_pair_count"]),
        "full_rho_quantiles_evaluable": finite_quantiles(
            stability.loc[stability["full_evaluable"], "full_rho_gr"]
        ),
        "tail_rho_quantiles_evaluable": finite_quantiles(
            stability.loc[stability["tail_evaluable"], "tail_rho_gr"]
        ),
        "full_horizontal_std_quantiles": finite_quantiles(
            stability["full_horizontal_std"]
        ),
        "full_typewell_std_quantiles": finite_quantiles(
            stability["full_typewell_std"]
        ),
        "full_mean_bias_quantiles": finite_quantiles(stability["full_mean_bias"]),
        "full_normalized_bias_quantiles": finite_quantiles(
            stability["full_normalized_bias"]
        ),
        "full_tail_rho_delta_quantiles_joint": finite_quantiles(
            joint["full_tail_rho_delta"]
        ),
    }
    return stability, fold_metrics, pooled


def evaluate_stage_0_gate(
    pooled: Mapping[str, Any],
    fold_metrics: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    spec = dict(get_nested(config, "model.stage_0.pass_requires_all", {}))
    poor_lower, poor_upper = [
        float(value) for value in spec["poor_multiplier_well_fraction_range"]
    ]
    poor_fraction = float(pooled["poor_multiplier_fraction"])
    spearman = float(pooled["full_tail_spearman_correlation"])
    multiplier_agreement = float(pooled["full_tail_multiplier_agreement"])
    minimum_fold_coverage = float(pooled["minimum_fold_full_evaluable_fraction"])
    checks = {
        "expected_well_count": int(pooled["actual_wells"])
        == int(pooled["expected_wells"]),
        "expected_fold_identity": list(pooled["actual_folds"])
        == list(pooled["expected_folds"]),
        "minimum_primary_evaluable_well_fraction": float(
            pooled["full_evaluable_fraction"]
        )
        >= float(spec["minimum_primary_evaluable_well_fraction"]),
        "maximum_fallback_well_fraction": float(pooled["fallback_fraction"])
        <= float(spec["maximum_fallback_well_fraction"]),
        "poor_multiplier_well_fraction_range": poor_lower
        <= poor_fraction
        <= poor_upper,
        "minimum_each_fold_primary_evaluable_fraction": (
            np.isfinite(minimum_fold_coverage)
            and minimum_fold_coverage
            >= float(spec["minimum_each_fold_primary_evaluable_fraction"])
        ),
        "minimum_tail_evaluable_well_fraction": float(
            pooled["tail_evaluable_fraction"]
        )
        >= float(spec["minimum_tail_evaluable_well_fraction"]),
        "minimum_full_tail_multiplier_agreement": (
            np.isfinite(multiplier_agreement)
            and multiplier_agreement
            >= float(spec["minimum_full_tail_multiplier_agreement"])
        ),
        "minimum_full_tail_spearman_correlation": (
            np.isfinite(spearman)
            and spearman
            >= float(spec["minimum_full_tail_spearman_correlation"])
        ),
    }
    if len(fold_metrics) != len(pooled["expected_folds"]):
        checks["expected_fold_identity"] = False
    passed = bool(all(checks.values()))
    return {
        "passed": passed,
        "stage_1_eligible": passed,
        "checks": {key: bool(value) for key, value in checks.items()},
        "thresholds": spec,
        "observed": to_jsonable(dict(pooled)),
        "decision": (
            "stage_0_passed_stage_1_requires_separate_user_approval"
            if passed
            else "stage_0_failed_close_without_rescue"
        ),
        "stage_1_implemented": False,
        "stage_1_executed": False,
        "rescue_grid_allowed": False,
    }


# %% [markdown]
# ## 7. Stage 0 orchestration and generated artifacts

# %%
def run_stage_0_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_scientific_contract(config, require_run_approval=True)
    started = time.perf_counter()
    output = artifact_dir()
    ledger = TargetFreeLedger()

    raw_dir = resolve_raw_train_dir(config)
    raw_identity, raw_manifest = validate_raw_well_identity(config, raw_dir)
    fold_assignment, fold_manifest = load_fold_assignment(config)
    raw_wells = raw_identity["well_id"].astype(str).tolist()
    fold_wells = fold_assignment["well_id"].astype(str).tolist()
    if raw_wells != fold_wells:
        missing = sorted(set(fold_wells) - set(raw_wells))
        extra = sorted(set(raw_wells) - set(fold_wells))
        raise ValueError(
            f"raw/fold well identity mismatch: missing={missing} extra={extra}"
        )
    fold_by_well = (
        fold_assignment.set_index("well_id")["fold"].astype(int).to_dict()
    )
    raw_sha_by_well = raw_identity.set_index("well_id").to_dict("index")

    agreement_rows = []
    well_manifest_rows = []
    for index, well in enumerate(raw_wells, start=1):
        horizontal, typewell, horizontal_path, typewell_path = load_target_free_well(
            raw_dir,
            well,
            ledger,
        )
        agreement_rows.append(
            build_well_agreement(
                horizontal,
                typewell,
                well_id=well,
                fold=int(fold_by_well[well]),
                config=config,
            )
        )
        well_manifest_rows.append(
            {
                "well_id": well,
                "fold": int(fold_by_well[well]),
                "horizontal_rows": int(len(horizontal)),
                "known_prefix_raw_rows": int(
                    pd.to_numeric(horizontal["TVT_input"], errors="coerce")
                    .notna()
                    .sum()
                ),
                "typewell_rows": int(len(typewell)),
                "horizontal_path": str(horizontal_path),
                "typewell_path": str(typewell_path),
                **raw_sha_by_well[well],
            }
        )
        if index % 50 == 0 or index == len(raw_wells):
            print(f"exp397 Stage 0 progress: {index}/{len(raw_wells)} wells")

    agreement_schedule = (
        pd.DataFrame(agreement_rows)
        .sort_values("well_id", kind="mergesort")
        .reset_index(drop=True)
    )
    freeze = freeze_target_free_agreement(agreement_schedule, ledger, config)
    verify_target_free_freeze(agreement_schedule, freeze)
    coefficient_table = agreement_schedule[COEFFICIENT_COLUMNS].copy()
    stability, fold_metrics, pooled = build_stability_readout(
        agreement_schedule,
        config,
    )
    gate = evaluate_stage_0_gate(pooled, fold_metrics, config)
    verify_target_free_freeze(agreement_schedule, freeze)

    scientific_contract = build_scientific_contract(config)
    input_manifest = {
        "raw_train": raw_manifest,
        "reporting_fold": fold_manifest,
        "raw_and_fold_well_identity_match": True,
        "horizontal_columns_read": SAFE_HORIZONTAL_COLUMNS,
        "horizontal_truth_columns_read": [],
        "hidden_like_assignment_loaded": False,
        "exp209_control_loaded": False,
        "saved_likpf_loaded": False,
        "hmm_executed": False,
        "input_manifest_sha256": "",
    }
    input_manifest["input_manifest_sha256"] = mapping_sha256(
        {key: value for key, value in input_manifest.items() if key != "input_manifest_sha256"}
    )

    manifests = {
        "agreement_schedule": write_frame(
            agreement_schedule,
            output / f"{OUTPUT_PREFIX}_agreement_schedule.csv.gz",
            sort_columns=["well_id"],
        ),
        "coefficient_table": write_frame(
            coefficient_table,
            output / f"{OUTPUT_PREFIX}_coefficient_table.csv",
            sort_columns=["well_id"],
        ),
        "stability_readout": write_frame(
            stability,
            output / f"{OUTPUT_PREFIX}_stability_readout.csv",
            sort_columns=["well_id"],
        ),
        "fold_metrics": write_frame(
            fold_metrics,
            output / f"{OUTPUT_PREFIX}_fold_metrics.csv",
            sort_columns=["fold"],
        ),
        "well_manifest": write_frame(
            pd.DataFrame(well_manifest_rows),
            output / f"{OUTPUT_PREFIX}_well_manifest.csv",
            sort_columns=["well_id"],
        ),
    }
    json_manifests = {
        "gate": write_json(output / f"{OUTPUT_PREFIX}_gate.json", gate),
        "input_manifest": write_json(
            output / f"{OUTPUT_PREFIX}_input_manifest.json",
            input_manifest,
        ),
        "scientific_contract": write_json(
            output / f"{OUTPUT_PREFIX}_scientific_contract.json",
            scientific_contract,
        ),
        "target_free_freeze": write_json(
            output / f"{OUTPUT_PREFIX}_target_free_freeze.json",
            freeze,
        ),
    }
    runtime_seconds = float(time.perf_counter() - started)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "stage_0_passed" if gate["passed"] else "stage_0_failed_closed",
        "route": "pf_beam",
        "stage": "stage_0_prefix_gr_agreement_stability",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "runtime_seconds": runtime_seconds,
        "runtime_versions": runtime_versions(),
        "execution": get_nested(config, "execution_contract.stage_0"),
        "pooled": pooled,
        "gate": gate,
        "freeze": freeze,
        "input_manifest": input_manifest,
        "scientific_contract": scientific_contract,
        "generated_artifacts": {**manifests, **json_manifests},
        "stage_1_implemented": False,
        "stage_1_executed": False,
        "inference_enabled": False,
        "submission_created": False,
    }
    summary_manifest = write_json(
        output / f"{OUTPUT_PREFIX}_summary.json",
        summary,
    )
    summary["generated_artifacts"]["summary"] = summary_manifest
    write_json(metrics_output_path(), summary)
    return summary


# %% [markdown]
# ## 8. Setup and fixed contract checks

# %%
CONFIG = load_experiment_config()
STAGE_0_EXECUTION_CONTRACT = validate_scientific_contract(CONFIG)
SCIENTIFIC_CONTRACT = build_scientific_contract(CONFIG)

print(
    json.dumps(
        {
            "experiment": EXPERIMENT_NAME,
            "route": get_nested(CONFIG, "experiment.route"),
            "status": get_nested(CONFIG, "experiment.status"),
            "implementation_scope": get_nested(CONFIG, "implementation.scope"),
            "stage_0_execution": STAGE_0_EXECUTION_CONTRACT,
            "scientific_contract_sha256": SCIENTIFIC_CONTRACT[
                "scientific_contract_sha256"
            ],
            "kaggle_push_approved": get_nested(
                CONFIG,
                "execution.kaggle_push_approved",
            ),
            "train_run_approved": get_nested(
                CONFIG,
                "execution.train_run_approved",
            ),
            "run_stage_0": get_nested(CONFIG, "execution.run_stage_0"),
            "stage_1_implemented": get_nested(
                CONFIG,
                "implementation.stage_1_implemented",
            ),
        },
        indent=2,
    )
)

# %% [markdown]
# ## 9. Fail-closed Stage 0 execution
#
# 初回のfull Stage 0は Kaggle private CPUを正とする。package/push/runの全承認flagが
# 明示的に有効でない限り、このcellは生成物を作らず停止する。Stage 0 PASSでもStage 1へは
# 自動進行しない。

# %%
if EXECUTE_NOTEBOOK:
    STAGE_0_SUMMARY = run_stage_0_experiment(CONFIG)
    display(pd.DataFrame([STAGE_0_SUMMARY["pooled"]]))
    display(pd.DataFrame([STAGE_0_SUMMARY["gate"]["checks"]]))
    print(json.dumps(to_jsonable(STAGE_0_SUMMARY["gate"]), indent=2))
else:
    STAGE_0_SUMMARY = None
