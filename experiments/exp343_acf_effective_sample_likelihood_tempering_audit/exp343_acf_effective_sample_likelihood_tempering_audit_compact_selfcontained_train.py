# %% [markdown]
# # exp343 ACF effective-sample likelihood tempering audit — Stage 0
#
# known-prefix の raw finite GR residual だけから full-prefix / last-512-prefix の
# integrated autocorrelation time を推定する。Stage 0 は tau schedule の安定性監査だけを
# 行い、HMM、suffix truth、prediction、inference、submission は扱わない。

# %% [markdown]
# ## Contents
# 1. Imports and notebook-safe configuration
# 2. Runtime, SHA, and deterministic output helpers
# 3. Scientific and execution contract
# 4. Fixed fold input and raw-well loading
# 5. Known-prefix residual and contiguous-run construction
# 6. ACF, outer-train prior, and tau schedule
# 7. Stability metrics and fail-closed gate
# 8. Stage 0 orchestration and generated artifacts
# 9. Setup and fixed input checks
# 10. Run Stage 0 and display the decision

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    from IPython import get_ipython
    from IPython.display import display
except ImportError:  # pragma: no cover - Kaggle and the repo environment provide IPython.
    get_ipython = lambda: None

    def display(value: Any) -> None:
        print(value)


EXPERIMENT_NAME = "exp343_acf_effective_sample_likelihood_tempering_audit"
OUTPUT_PREFIX = EXPERIMENT_NAME
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
IMPORT_ONLY = os.environ.get("EXP343_IMPORT_ONLY", "0") == "1"
EXECUTE_NOTEBOOK = get_ipython() is not None and not IMPORT_ONLY


def read_yaml(path: Path) -> dict[str, Any]:
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
    for candidate in (PACKAGE_DIR, *PACKAGE_DIR.parents):
        if (candidate / "project.yml").is_file():
            return candidate
    return PACKAGE_DIR


def load_experiment_config() -> dict[str, Any]:
    root = project_root()
    candidates = (
        PACKAGE_DIR / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    )
    for path in candidates:
        if not path.is_file():
            continue
        config = read_yaml(path)
        if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
            return config
    raise FileNotFoundError(f"Could not locate exp343 config; checked={candidates}")


CONFIG = load_experiment_config()

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
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_gzip_decompressed(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mapping_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        to_jsonable(dict(payload)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def dataframe_content_sha(frame: pd.DataFrame, sort_columns: Sequence[str]) -> str:
    ordered = frame.sort_values(list(sort_columns), kind="mergesort").reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update(json.dumps(list(ordered.columns), separators=(",", ":")).encode())
    digest.update(json.dumps([str(dtype) for dtype in ordered.dtypes]).encode())
    hashed = pd.util.hash_pandas_object(
        ordered,
        index=False,
        categorize=True,
    ).to_numpy(np.uint64)
    digest.update(hashed.tobytes())
    return digest.hexdigest()


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
        "filename": path.name,
        "rows": int(len(ordered)),
        "columns": list(ordered.columns),
        "raw_sha256": sha256_path(path),
        "content_sha256": dataframe_content_sha(ordered, sort_columns),
    }
    if path.suffix == ".gz":
        manifest["decompressed_sha256"] = sha256_gzip_decompressed(path)
    return manifest


def artifact_dir() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        output = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        output = project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    output.mkdir(parents=True, exist_ok=True)
    return output


def runtime_is_kaggle() -> bool:
    return KAGGLE_INPUT_ROOT.exists() and KAGGLE_WORKING_ROOT.exists()


# %% [markdown]
# ## 3. Scientific and execution contract
#
# Stage 0 は 1 deterministic diagnostic、5 reporting folds、0 HMM、0 model config、
# 0 trained fold、0 booster。Stage 1 の decoder はこの notebook に存在しない。

# %%
def validate_scientific_contract(
    config: Mapping[str, Any],
    *,
    require_run_approval: bool = False,
) -> dict[str, Any]:
    if get_nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp343 route must remain pf_beam")
    if get_nested(config, "lineage.parent") != (
        "exp281_exp226_residual_offset_exact_hmm_transition_probe"
    ):
        raise ValueError("exp343 scientific parent must remain exp281")
    if not bool(get_nested(config, "implementation.enabled")):
        raise ValueError("Stage 0 implementation must be enabled")
    if bool(get_nested(config, "implementation.stage_1_implemented")):
        raise ValueError("Stage 1 must remain unimplemented")
    acf = dict(get_nested(config, "model.acf") or {})
    expected_acf = {
        "lags": [1, 20],
        "minimum_finite_residuals": 128,
        "minimum_pairs_each_lag": 20,
        "forbid_cross_missing_pairs": True,
        "shrinkage_support_k": 200,
        "shrinkage_space": "log",
        "shrinkage_prior": "outer_train_fold_median",
        "tau_clip": [1.0, 4.0],
        "insufficient_fallback": "outer_train_fold_median",
        "rho_estimator": "pairwise_pearson_within_contiguous_finite_runs",
    }
    for key, expected in expected_acf.items():
        if acf.get(key) != expected:
            raise ValueError(f"fixed ACF contract changed at {key}: {acf.get(key)}")
    if acf.get("tau_raw") != "one_plus_two_sum_positive_acf":
        raise ValueError("tau_raw formula must remain fixed")
    if list(get_nested(config, "model.stage_0.windows") or []) != [
        "full_known_prefix",
        "last_512_known_prefix_rows",
    ]:
        raise ValueError("Stage 0 windows must remain full prefix and last 512 prefix rows")
    stage_0 = dict(get_nested(config, "execution_contract.stage_0") or {})
    expected_stage_0 = {
        "diagnostic_variants": 1,
        "hmm_well_runs": 0,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
    }
    if stage_0 != expected_stage_0:
        raise ValueError(f"Stage 0 execution contract changed: {stage_0}")
    if bool(get_nested(config, "execution.run_stage_1")):
        raise ValueError("Stage 1 is not implemented and cannot run")
    if bool(get_nested(config, "execution.run_inference")):
        raise ValueError("Inference must remain disabled")
    if bool(get_nested(config, "execution.create_submission")):
        raise ValueError("Submission must remain disabled")
    if bool(get_nested(config, "execution_contract.parent_control_retraining")):
        raise ValueError("Parent/control regeneration is forbidden")
    if bool(get_nested(config, "runtime.kaggle.enable_gpu")):
        raise ValueError("Stage 0 must run on CPU")
    if bool(get_nested(config, "runtime.kaggle.enable_internet")):
        raise ValueError("Stage 0 must run with internet disabled")
    if require_run_approval:
        approved = bool(get_nested(config, "execution.kaggle_push_approved"))
        enabled = bool(get_nested(config, "execution.run_stage_0"))
        run_on_push = bool(get_nested(config, "runtime.kaggle.train_run_on_push"))
        if not (approved and enabled and run_on_push):
            raise RuntimeError(
                "Stage 0 Kaggle package/push/run is not approved; all three approval "
                "flags must be true for execution"
            )
        if not runtime_is_kaggle():
            raise RuntimeError("The first full Stage 0 run must execute on Kaggle CPU")
    return stage_0


# %% [markdown]
# ## 4. Fixed fold input and raw-well loading
#
# exp226 OOF からは `well_id` と group-safe `fold` だけを読む。suffix prediction と
# truth columns は読み込まない。raw horizontal も `TVT` を reader で除外する。

# %%
def resolve_existing(filename: str, candidates: Sequence[str]) -> Path:
    checked: list[str] = []
    root = project_root()
    for raw in candidates:
        candidate = Path(str(raw))
        for path in (candidate, root / candidate, PACKAGE_DIR / candidate):
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
    raise FileNotFoundError(f"Could not resolve {filename}; checked={checked[:50]}")


def resolve_raw_train_dir(config: Mapping[str, Any]) -> Path:
    candidates = [
        Path(str(value))
        for value in (get_nested(config, "data.raw_train_candidates") or [])
    ]
    local = project_root() / str(get_nested(config, "data.train_dir") or "data/raw/train")
    candidates.append(local)
    for path in candidates:
        if path.is_dir() and next(path.glob("*__horizontal_well.csv"), None) is not None:
            return path
    if KAGGLE_INPUT_ROOT.exists():
        for path in sorted(KAGGLE_INPUT_ROOT.rglob("train")):
            if path.is_dir() and next(path.glob("*__horizontal_well.csv"), None) is not None:
                return path
    raise FileNotFoundError("Could not resolve the raw competition train directory")


def load_fold_assignment(
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    spec = dict(get_nested(config, "data.exp226_oof") or {})
    path = resolve_existing(
        str(spec["filename"]),
        [str(value) for value in spec.get("candidates", [])],
    )
    actual = sha256_gzip_decompressed(path)
    expected = str(spec["expected_decompressed_sha256"])
    if actual != expected:
        raise ValueError(
            f"exp226 OOF decompressed SHA mismatch: expected={expected} actual={actual}"
        )
    frame = pd.read_csv(
        path,
        usecols=["well_id", "fold"],
        dtype={"well_id": str},
    )
    frame["fold"] = pd.to_numeric(frame["fold"], errors="raise").astype(np.int64)
    if len(frame) != int(get_nested(config, "validation.expected_rows")):
        raise ValueError("exp226 OOF row count does not match the fixed contract")
    grouped = frame.groupby("well_id", sort=True)["fold"].agg(["nunique", "first"])
    if not bool(grouped["nunique"].eq(1).all()):
        raise ValueError("each exp226 well must belong to exactly one fold")
    assignment = grouped["first"].rename("fold").reset_index()
    expected_wells = int(get_nested(config, "validation.expected_wells"))
    expected_folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    if len(assignment) != expected_wells:
        raise ValueError(f"exp226 fold assignment must contain {expected_wells} wells")
    if sorted(assignment["fold"].unique().tolist()) != expected_folds:
        raise ValueError("exp226 fold set does not match the fixed contract")
    manifest = {
        "name": "exp226_group_safe_fold_assignment",
        "path": str(path),
        "bytes": path.stat().st_size,
        "raw_sha256": sha256_path(path),
        "decompressed_sha256": actual,
        "rows": int(len(frame)),
        "wells": int(len(assignment)),
        "folds": expected_folds,
        "content_sha256": dataframe_content_sha(
            assignment,
            ["well_id"],
        ),
    }
    return assignment, path, manifest


def list_raw_wells(raw_dir: Path) -> list[str]:
    wells = []
    for horizontal_path in sorted(raw_dir.glob("*__horizontal_well.csv")):
        well = horizontal_path.name.removesuffix("__horizontal_well.csv")
        if (raw_dir / f"{well}__typewell.csv").is_file():
            wells.append(well)
    return wells


def load_target_free_well(
    raw_dir: Path,
    well: str,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    horizontal_path = raw_dir / f"{well}__horizontal_well.csv"
    typewell_path = raw_dir / f"{well}__typewell.csv"
    horizontal = pd.read_csv(
        horizontal_path,
        usecols=lambda column: column != "TVT",
    )
    if "TVT" in horizontal.columns:
        raise ValueError("target-free horizontal reader exposed suffix truth")
    required_horizontal = {"GR", "TVT_input"}
    if not required_horizontal.issubset(horizontal.columns):
        raise ValueError(
            f"{well} horizontal missing {sorted(required_horizontal - set(horizontal.columns))}"
        )
    typewell = pd.read_csv(typewell_path, usecols=["TVT", "GR"])
    return horizontal, typewell, horizontal_path, typewell_path


# %% [markdown]
# ## 5. Known-prefix residual and contiguous-run construction
#
# last-512 は finite residual を512個取るのではなく、known-prefix の末尾512 raw rowsを
# 先に固定し、その中の finite residual だけを使う。run は raw row index が1ずつ連続する
# 範囲で分割するため、GR missing をまたぐ lag pair は生成されない。

# %%
def prepare_typewell(typewell: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    frame = typewell[["TVT", "GR"]].copy()
    frame["TVT"] = pd.to_numeric(frame["TVT"], errors="coerce")
    frame["GR"] = pd.to_numeric(frame["GR"], errors="coerce")
    frame = frame.dropna(subset=["TVT"]).sort_values("TVT", kind="mergesort")
    frame["GR"] = frame["GR"].ffill().bfill()
    values = frame[["TVT", "GR"]].to_numpy(np.float64)
    if len(frame) < 2 or not np.isfinite(values).all():
        raise ValueError("typewell requires at least two finite TVT/GR rows")
    return values[:, 0], values[:, 1]


def assign_contiguous_runs(row_idx: np.ndarray) -> np.ndarray:
    if len(row_idx) == 0:
        return np.empty(0, dtype=np.int64)
    breaks = np.r_[True, np.diff(row_idx) != 1]
    return np.cumsum(breaks).astype(np.int64) - 1


def build_known_residuals(
    horizontal_without_truth: pd.DataFrame,
    typewell: pd.DataFrame,
    *,
    well_id: str,
    fold: int,
    tail_rows: int = 512,
) -> pd.DataFrame:
    if "TVT" in horizontal_without_truth.columns:
        raise ValueError("known-prefix residual construction forbids horizontal TVT")
    tvt_input = pd.to_numeric(
        horizontal_without_truth["TVT_input"],
        errors="coerce",
    ).to_numpy(np.float64)
    observed_gr = pd.to_numeric(
        horizontal_without_truth["GR"],
        errors="coerce",
    ).to_numpy(np.float64)
    known_positions = np.flatnonzero(np.isfinite(tvt_input))
    if len(known_positions) == 0:
        return pd.DataFrame(
            columns=[
                "well_id",
                "fold",
                "row_idx",
                "known_position",
                "known_rows",
                "in_last_512",
                "tvt_input",
                "observed_gr",
                "typewell_gr",
                "residual",
                "full_run_id",
                "last_512_run_id",
            ]
        )
    typewell_tvt, typewell_gr = prepare_typewell(typewell)
    expected_gr = np.interp(tvt_input[known_positions], typewell_tvt, typewell_gr)
    finite = np.isfinite(observed_gr[known_positions]) & np.isfinite(expected_gr)
    finite_positions = known_positions[finite]
    known_ordinal = np.arange(len(known_positions), dtype=np.int64)[finite]
    tail_start = max(0, len(known_positions) - int(tail_rows))
    in_tail = known_ordinal >= tail_start
    full_runs = assign_contiguous_runs(finite_positions)
    tail_runs = np.full(len(finite_positions), -1, dtype=np.int64)
    tail_runs[in_tail] = assign_contiguous_runs(finite_positions[in_tail])
    frame = pd.DataFrame(
        {
            "well_id": str(well_id),
            "fold": int(fold),
            "row_idx": finite_positions.astype(np.int64),
            "known_position": known_ordinal,
            "known_rows": int(len(known_positions)),
            "in_last_512": in_tail,
            "tvt_input": tvt_input[finite_positions],
            "observed_gr": observed_gr[finite_positions],
            "typewell_gr": expected_gr[finite],
            "residual": observed_gr[finite_positions] - expected_gr[finite],
            "full_run_id": full_runs,
            "last_512_run_id": tail_runs,
        }
    )
    if not np.isfinite(
        frame[["tvt_input", "observed_gr", "typewell_gr", "residual"]].to_numpy(
            np.float64
        )
    ).all():
        raise ValueError(f"{well_id} residual schedule contains non-finite values")
    return frame


# %% [markdown]
# ## 6. ACF, outer-train prior, and tau schedule
#
# 各 lag の rho は、contiguous finite run 内の有効 pair を連結した pairwise Pearson
# correlation。全 lag が20 pair以上かつ finite、window全体が128 residual以上の場合だけ
# raw tau を evaluable とする。

# %%
def pairwise_pearson(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) != len(right) or len(left) < 2:
        return float("nan")
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = float(
        np.sqrt(np.dot(left_centered, left_centered) * np.dot(right_centered, right_centered))
    )
    if not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    return float(np.dot(left_centered, right_centered) / denominator)


def estimate_window_acf(
    residuals: pd.DataFrame,
    *,
    window: str,
    lags: Sequence[int],
    minimum_finite_residuals: int,
    minimum_pairs_each_lag: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if window == "full_known_prefix":
        selected = residuals.copy()
        run_column = "full_run_id"
    elif window == "last_512_known_prefix_rows":
        selected = residuals.loc[residuals["in_last_512"]].copy()
        run_column = "last_512_run_id"
    else:
        raise ValueError(f"unknown Stage 0 window: {window}")
    selected = selected.sort_values("row_idx", kind="mergesort")
    lag_rows = []
    for lag in lags:
        left_parts: list[np.ndarray] = []
        right_parts: list[np.ndarray] = []
        for _, run in selected.groupby(run_column, sort=True):
            values = run["residual"].to_numpy(np.float64)
            if len(values) <= int(lag):
                continue
            left_parts.append(values[:-int(lag)])
            right_parts.append(values[int(lag):])
        if left_parts:
            left = np.concatenate(left_parts)
            right = np.concatenate(right_parts)
        else:
            left = np.empty(0, dtype=np.float64)
            right = np.empty(0, dtype=np.float64)
        rho = pairwise_pearson(left, right)
        lag_rows.append(
            {
                "window": window,
                "lag": int(lag),
                "pair_count": int(len(left)),
                "rho": rho,
                "positive_rho": max(rho, 0.0) if np.isfinite(rho) else np.nan,
            }
        )
    lag_frame = pd.DataFrame(lag_rows)
    sufficient_n = len(selected) >= int(minimum_finite_residuals)
    sufficient_pairs = bool(
        lag_frame["pair_count"].ge(int(minimum_pairs_each_lag)).all()
    )
    finite_rho = bool(np.isfinite(lag_frame["rho"]).all())
    evaluable = bool(sufficient_n and sufficient_pairs and finite_rho)
    tau_raw = (
        float(1.0 + 2.0 * lag_frame["positive_rho"].sum())
        if evaluable
        else float("nan")
    )
    if evaluable and (not np.isfinite(tau_raw) or tau_raw < 1.0):
        raise ValueError("evaluable tau_raw must be finite and at least one")
    metadata = {
        "window": window,
        "finite_residual_count": int(len(selected)),
        "contiguous_run_count": int(selected[run_column].nunique()),
        "minimum_pair_count": int(lag_frame["pair_count"].min()),
        "lag_1_rho": float(lag_frame.loc[lag_frame["lag"].eq(1), "rho"].iloc[0]),
        "evaluable": evaluable,
        "fallback_reason": (
            ""
            if evaluable
            else "|".join(
                reason
                for reason, failed in (
                    ("finite_residual_count", not sufficient_n),
                    ("lag_pair_count", not sufficient_pairs),
                    ("nonfinite_rho", not finite_rho),
                )
                if failed
            )
        ),
        "tau_raw": tau_raw,
    }
    return lag_frame, metadata


def estimate_well_acf(
    residuals: pd.DataFrame,
    config: Mapping[str, Any],
    *,
    well_id: str | None = None,
    fold: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if residuals.empty:
        if well_id is None or fold is None:
            raise ValueError("empty residual input requires explicit well_id and fold")
        current_well = str(well_id)
        current_fold = int(fold)
    else:
        current_well = str(residuals["well_id"].iloc[0])
        current_fold = int(residuals["fold"].iloc[0])
    acf = dict(get_nested(config, "model.acf") or {})
    lag_min, lag_max = [int(value) for value in acf["lags"]]
    lags = list(range(lag_min, lag_max + 1))
    tau_rows = []
    lag_frames = []
    for window in get_nested(config, "model.stage_0.windows"):
        lag_frame, metadata = estimate_window_acf(
            residuals,
            window=str(window),
            lags=lags,
            minimum_finite_residuals=int(acf["minimum_finite_residuals"]),
            minimum_pairs_each_lag=int(acf["minimum_pairs_each_lag"]),
        )
        lag_frame.insert(0, "fold", current_fold)
        lag_frame.insert(0, "well_id", current_well)
        lag_frames.append(lag_frame)
        tau_rows.append(
            {
                "well_id": current_well,
                "fold": current_fold,
                **metadata,
            }
        )
    return pd.concat(lag_frames, ignore_index=True), pd.DataFrame(tau_rows)


def attach_outer_train_tau_prior(
    raw_tau: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    acf = dict(get_nested(config, "model.acf") or {})
    folds = [int(value) for value in get_nested(config, "validation.expected_folds")]
    windows = [str(value) for value in get_nested(config, "model.stage_0.windows")]
    prior_rows = []
    for window in windows:
        window_rows = raw_tau.loc[raw_tau["window"].eq(window)]
        for fold in folds:
            outer_train = window_rows.loc[
                window_rows["fold"].ne(fold) & window_rows["evaluable"]
            ]
            values = outer_train["tau_raw"].to_numpy(np.float64)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                raise ValueError(f"no evaluable outer-train tau for window={window} fold={fold}")
            prior_rows.append(
                {
                    "window": window,
                    "fold": fold,
                    "tau_fold_median": float(np.median(values)),
                    "prior_source_wells": int(len(values)),
                }
            )
    priors = pd.DataFrame(prior_rows)
    schedule = raw_tau.merge(
        priors,
        on=["window", "fold"],
        how="left",
        validate="many_to_one",
    )
    if schedule["tau_fold_median"].isna().any():
        raise ValueError("tau schedule is missing an outer-train fold prior")
    support_k = float(acf["shrinkage_support_k"])
    lower, upper = [float(value) for value in acf["tau_clip"]]
    evaluable = schedule["evaluable"].to_numpy(bool)
    n = schedule["finite_residual_count"].to_numpy(np.float64)
    prior = schedule["tau_fold_median"].to_numpy(np.float64)
    raw = schedule["tau_raw"].to_numpy(np.float64)
    alpha = np.where(evaluable, n / (n + support_k), 0.0)
    base = np.where(evaluable, raw, prior)
    shrunk = np.exp(alpha * np.log(base) + (1.0 - alpha) * np.log(prior))
    schedule["fallback"] = ~evaluable
    schedule["alpha"] = alpha
    schedule["tau_shrunk"] = shrunk
    schedule["tau_eff"] = np.clip(shrunk, lower, upper)
    schedule["upper_clipped"] = shrunk > upper
    schedule["emission_multiplier_stage_1_if_approved"] = (
        1.0 / schedule["tau_eff"].to_numpy(np.float64)
    )
    required = [
        "tau_fold_median",
        "alpha",
        "tau_shrunk",
        "tau_eff",
        "emission_multiplier_stage_1_if_approved",
    ]
    if not np.isfinite(schedule[required].to_numpy(np.float64)).all():
        raise ValueError("tau schedule contains non-finite required values")
    return schedule, priors


# %% [markdown]
# ## 7. Stability metrics and fail-closed gate
#
# full/last-512 stability は両windowがraw-evaluableな well だけで測る。fallback prior が
# 同一であることによる見かけの相関で gate を通さない。coverage / fallback は773 wellsを
# 分母とし、window別 median / clip / fold ratio は悪い側で判定する。

# %%
def spearman_rank_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(left_array) & np.isfinite(right_array)
    if int(finite.sum()) < 2:
        return float("nan")
    left_rank = pd.Series(left_array[finite]).rank(method="average").to_numpy(np.float64)
    right_rank = pd.Series(right_array[finite]).rank(method="average").to_numpy(np.float64)
    return pairwise_pearson(left_rank, right_rank)


def build_stability_readout(
    schedule: pd.DataFrame,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    full_name = "full_known_prefix"
    tail_name = "last_512_known_prefix_rows"
    fields = [
        "well_id",
        "fold",
        "finite_residual_count",
        "evaluable",
        "fallback",
        "tau_raw",
        "tau_fold_median",
        "alpha",
        "tau_shrunk",
        "tau_eff",
        "upper_clipped",
    ]
    full = schedule.loc[schedule["window"].eq(full_name), fields].copy()
    tail = schedule.loc[schedule["window"].eq(tail_name), fields].copy()
    full = full.rename(
        columns={column: f"full_{column}" for column in fields if column not in {"well_id", "fold"}}
    )
    tail = tail.rename(
        columns={column: f"tail_{column}" for column in fields if column not in {"well_id", "fold"}}
    )
    stability = full.merge(tail, on=["well_id", "fold"], how="inner", validate="one_to_one")
    stability["joint_evaluable"] = (
        stability["full_evaluable"].astype(bool) & stability["tail_evaluable"].astype(bool)
    )
    stability["any_fallback"] = (
        stability["full_fallback"].astype(bool) | stability["tail_fallback"].astype(bool)
    )
    stability["absolute_log_tau_ratio"] = np.abs(
        np.log(stability["full_tau_eff"].to_numpy(np.float64))
        - np.log(stability["tail_tau_eff"].to_numpy(np.float64))
    )
    gate_spec = dict(get_nested(config, "model.stage_0.pass_requires_all") or {})
    fold_rows = []
    for fold, part in stability.groupby("fold", sort=True):
        eligible = part.loc[part["joint_evaluable"]]
        rho = spearman_rank_correlation(
            eligible["full_tau_eff"],
            eligible["tail_tau_eff"],
        )
        median_log_ratio = (
            float(eligible["absolute_log_tau_ratio"].median())
            if not eligible.empty
            else float("nan")
        )
        stable = bool(
            np.isfinite(rho)
            and rho >= float(gate_spec["minimum_pooled_spearman_full_vs_tail"])
            and np.isfinite(median_log_ratio)
            and median_log_ratio
            <= float(gate_spec["maximum_median_absolute_log_ratio"])
        )
        fold_rows.append(
            {
                "fold": int(fold),
                "wells": int(len(part)),
                "joint_evaluable_wells": int(len(eligible)),
                "joint_evaluable_fraction": float(part["joint_evaluable"].mean()),
                "fallback_fraction": float(part["any_fallback"].mean()),
                "spearman_full_vs_tail": rho,
                "median_absolute_log_ratio": median_log_ratio,
                "full_median_tau_eff": float(part["full_tau_eff"].median()),
                "tail_median_tau_eff": float(part["tail_tau_eff"].median()),
                "stable": stable,
            }
        )
    fold_metrics = pd.DataFrame(fold_rows)
    eligible = stability.loc[stability["joint_evaluable"]]
    window_metrics: dict[str, dict[str, Any]] = {}
    for label in ("full", "tail"):
        fold_medians = fold_metrics[f"{label}_median_tau_eff"].to_numpy(np.float64)
        fold_ratio = float(np.max(fold_medians) / np.min(fold_medians))
        window_metrics[label] = {
            "median_tau_eff": float(stability[f"{label}_tau_eff"].median()),
            "upper_clip_fraction": float(stability[f"{label}_upper_clipped"].mean()),
            "fold_median_tau_max_min_ratio": fold_ratio,
        }
    pooled = {
        "expected_wells": int(get_nested(config, "validation.expected_wells")),
        "actual_wells": int(len(stability)),
        "joint_evaluable_wells": int(len(eligible)),
        "joint_evaluable_fraction": float(stability["joint_evaluable"].mean()),
        "fallback_wells": int(stability["any_fallback"].sum()),
        "fallback_fraction": float(stability["any_fallback"].mean()),
        "spearman_full_vs_tail": spearman_rank_correlation(
            eligible["full_tau_eff"],
            eligible["tail_tau_eff"],
        ),
        "median_absolute_log_ratio": (
            float(eligible["absolute_log_tau_ratio"].median())
            if not eligible.empty
            else float("nan")
        ),
        "stable_folds": int(fold_metrics["stable"].sum()),
        "windows": window_metrics,
    }
    return stability, fold_metrics, pooled


def evaluate_stage_0_gate(
    pooled: Mapping[str, Any],
    fold_metrics: pd.DataFrame,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    spec = dict(get_nested(config, "model.stage_0.pass_requires_all") or {})
    window_metrics = dict(pooled["windows"])
    checks = {
        "expected_well_count": int(pooled["actual_wells"])
        == int(pooled["expected_wells"]),
        "expected_fold_count": len(fold_metrics)
        == int(get_nested(config, "validation.n_folds")),
        "minimum_evaluable_well_fraction": float(pooled["joint_evaluable_fraction"])
        >= float(spec["minimum_evaluable_well_fraction"]),
        "maximum_fallback_well_fraction": float(pooled["fallback_fraction"])
        <= float(spec["maximum_fallback_well_fraction"]),
        "minimum_pooled_spearman_full_vs_tail": (
            np.isfinite(float(pooled["spearman_full_vs_tail"]))
            and float(pooled["spearman_full_vs_tail"])
            >= float(spec["minimum_pooled_spearman_full_vs_tail"])
        ),
        "maximum_median_absolute_log_ratio": (
            np.isfinite(float(pooled["median_absolute_log_ratio"]))
            and float(pooled["median_absolute_log_ratio"])
            <= float(spec["maximum_median_absolute_log_ratio"])
        ),
        "minimum_folds_stable": int(pooled["stable_folds"])
        >= int(spec["minimum_folds_stable"]),
        "minimum_pooled_median_tau_both_windows": min(
            float(window_metrics["full"]["median_tau_eff"]),
            float(window_metrics["tail"]["median_tau_eff"]),
        )
        >= float(spec["minimum_pooled_median_tau"]),
        "maximum_upper_clip_fraction_either_window": max(
            float(window_metrics["full"]["upper_clip_fraction"]),
            float(window_metrics["tail"]["upper_clip_fraction"]),
        )
        <= float(spec["maximum_upper_clip_fraction"]),
        "maximum_fold_median_tau_ratio_either_window": max(
            float(window_metrics["full"]["fold_median_tau_max_min_ratio"]),
            float(window_metrics["tail"]["fold_median_tau_max_min_ratio"]),
        )
        <= float(spec["maximum_fold_median_tau_ratio"]),
    }
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
# ## 8. Stage 0 orchestration and generated artifacts

# %%
def run_stage_0_experiment(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_scientific_contract(config, require_run_approval=True)
    started = time.perf_counter()
    output_dir = artifact_dir()
    fold_assignment, exp226_path, fold_manifest = load_fold_assignment(config)
    raw_dir = resolve_raw_train_dir(config)
    raw_wells = list_raw_wells(raw_dir)
    expected_wells = sorted(fold_assignment["well_id"].astype(str).tolist())
    if raw_wells != expected_wells:
        missing = sorted(set(expected_wells) - set(raw_wells))
        extra = sorted(set(raw_wells) - set(expected_wells))
        raise ValueError(f"raw/exp226 well identity mismatch missing={missing} extra={extra}")
    fold_by_well = fold_assignment.set_index("well_id")["fold"].astype(int).to_dict()
    all_residuals = []
    all_lags = []
    all_raw_tau = []
    well_manifests = []
    for index, well in enumerate(raw_wells, start=1):
        horizontal, typewell, horizontal_path, typewell_path = load_target_free_well(
            raw_dir,
            well,
        )
        residuals = build_known_residuals(
            horizontal,
            typewell,
            well_id=well,
            fold=int(fold_by_well[well]),
            tail_rows=512,
        )
        lag_frame, tau_frame = estimate_well_acf(
            residuals,
            config,
            well_id=well,
            fold=int(fold_by_well[well]),
        )
        all_residuals.append(residuals)
        all_lags.append(lag_frame)
        all_raw_tau.append(tau_frame)
        well_manifests.append(
            {
                "well_id": well,
                "fold": int(fold_by_well[well]),
                "horizontal_rows": int(len(horizontal)),
                "known_prefix_rows": int(
                    pd.to_numeric(horizontal["TVT_input"], errors="coerce").notna().sum()
                ),
                "finite_known_residuals": int(len(residuals)),
                "horizontal_raw_sha256": sha256_path(horizontal_path),
                "typewell_raw_sha256": sha256_path(typewell_path),
            }
        )
        if index % 50 == 0 or index == len(raw_wells):
            print(f"Stage 0 ACF progress: {index}/{len(raw_wells)} wells")
    residual_frame = pd.concat(all_residuals, ignore_index=True)
    lag_frame = pd.concat(all_lags, ignore_index=True)
    raw_tau = pd.concat(all_raw_tau, ignore_index=True)
    schedule, priors = attach_outer_train_tau_prior(raw_tau, config)
    stability, fold_metrics, pooled = build_stability_readout(schedule, config)
    gate = evaluate_stage_0_gate(pooled, fold_metrics, config)
    contract = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "stage": "stage_0_acf_stability",
        "source_fold_assignment": "exp226_group_safe_oof",
        "target_free_columns": ["TVT_input", "GR", "typewell.TVT", "typewell.GR"],
        "forbidden_columns": ["horizontal.TVT", "tvt_true", "error", "abs_error"],
        "windows": list(get_nested(config, "model.stage_0.windows")),
        "acf": dict(get_nested(config, "model.acf") or {}),
        "stage_0_execution": dict(get_nested(config, "execution_contract.stage_0") or {}),
        "stage_1_implemented": False,
        "inference_enabled": False,
        "submission_enabled": False,
    }
    contract["scientific_contract_sha256"] = mapping_sha256(contract)
    manifests = {}
    manifests["known_prefix_residuals"] = write_frame(
        residual_frame,
        output_dir / f"{OUTPUT_PREFIX}_known_prefix_residuals.csv.gz",
        sort_columns=["well_id", "row_idx"],
    )
    manifests["acf_lag_readout"] = write_frame(
        lag_frame,
        output_dir / f"{OUTPUT_PREFIX}_acf_lag_readout.csv.gz",
        sort_columns=["well_id", "window", "lag"],
    )
    manifests["tau_schedule"] = write_frame(
        schedule,
        output_dir / f"{OUTPUT_PREFIX}_tau_schedule.csv.gz",
        sort_columns=["well_id", "window"],
    )
    manifests["fold_priors"] = write_frame(
        priors,
        output_dir / f"{OUTPUT_PREFIX}_fold_priors.csv",
        sort_columns=["window", "fold"],
    )
    manifests["stability_readout"] = write_frame(
        stability,
        output_dir / f"{OUTPUT_PREFIX}_stability_readout.csv",
        sort_columns=["well_id"],
    )
    manifests["fold_metrics"] = write_frame(
        fold_metrics,
        output_dir / f"{OUTPUT_PREFIX}_fold_metrics.csv",
        sort_columns=["fold"],
    )
    manifests["well_manifest"] = write_frame(
        pd.DataFrame(well_manifests),
        output_dir / f"{OUTPUT_PREFIX}_well_manifest.csv",
        sort_columns=["well_id"],
    )
    input_manifest = {
        "exp226": fold_manifest,
        "exp226_path": str(exp226_path),
        "raw_train_dir": str(raw_dir),
        "raw_well_count": int(len(raw_wells)),
        "raw_well_identity_sha256": hashlib.sha256(
            "\n".join(raw_wells).encode()
        ).hexdigest(),
    }
    write_json(output_dir / f"{OUTPUT_PREFIX}_input_manifest.json", input_manifest)
    write_json(output_dir / f"{OUTPUT_PREFIX}_scientific_contract.json", contract)
    write_json(output_dir / f"{OUTPUT_PREFIX}_gate.json", gate)
    runtime_seconds = float(time.perf_counter() - started)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "stage_0_passed" if gate["passed"] else "stage_0_failed_closed",
        "route": "pf_beam",
        "stage": "stage_0_acf_stability",
        "runtime_seconds": runtime_seconds,
        "pooled": pooled,
        "gate": gate,
        "execution": dict(get_nested(config, "execution_contract.stage_0") or {}),
        "scientific_contract_sha256": contract["scientific_contract_sha256"],
        "tau_schedule_content_sha256": manifests["tau_schedule"]["content_sha256"],
        "acf_lag_content_sha256": manifests["acf_lag_readout"]["content_sha256"],
        "residual_content_sha256": manifests["known_prefix_residuals"]["content_sha256"],
        "artifacts": manifests,
        "stage_1_implemented": False,
        "stage_1_executed": False,
        "inference_executed": False,
        "submission_created": False,
    }
    write_json(output_dir / f"{OUTPUT_PREFIX}_summary.json", summary)
    write_json(
        KAGGLE_WORKING_ROOT / "metrics.json"
        if KAGGLE_WORKING_ROOT.exists()
        else project_root() / "experiments" / EXPERIMENT_NAME / "metrics.stage0.local.json",
        summary,
    )
    return summary


# %% [markdown]
# ## 9. Setup and fixed input checks

# %%
if EXECUTE_NOTEBOOK:
    contract = validate_scientific_contract(CONFIG, require_run_approval=True)
    setup = {
        "experiment": EXPERIMENT_NAME,
        "parent": get_nested(CONFIG, "lineage.parent"),
        "route": get_nested(CONFIG, "experiment.route"),
        "active_stage": get_nested(CONFIG, "execution.active_stage"),
        "windows": get_nested(CONFIG, "model.stage_0.windows"),
        "acf": get_nested(CONFIG, "model.acf"),
        "stage_0_execution_contract": contract,
        "stage_1_implemented": get_nested(CONFIG, "implementation.stage_1_implemented"),
        "inference_enabled": get_nested(CONFIG, "inference.enabled"),
        "submission_enabled": get_nested(CONFIG, "inference.create_submission"),
        "runtime": get_nested(CONFIG, "runtime.kaggle"),
    }
    display(setup)
    print("Leakage policy")
    for rule in get_nested(CONFIG, "validation.leakage_policy"):
        print("-", rule)

# %% [markdown]
# ## 10. Run Stage 0 and display the decision

# %%
if EXECUTE_NOTEBOOK:
    STAGE_0_SUMMARY = run_stage_0_experiment(CONFIG)
    display(pd.DataFrame([STAGE_0_SUMMARY["pooled"]]).T)
    display(STAGE_0_SUMMARY["gate"])
    print(json.dumps(to_jsonable(STAGE_0_SUMMARY), indent=2, ensure_ascii=False))
