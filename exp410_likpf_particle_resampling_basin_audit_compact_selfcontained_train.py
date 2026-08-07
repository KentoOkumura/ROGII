# %% [markdown]
# # exp410 likelihood-PF particle / resampling basin audit — train
#
# Re-run the unchanged exp072 likelihood PF on preregistered PF persistent-offset
# wells. Capture particle support before GR, after GR, and after resampling without
# changing the random-number stream or producing a prediction candidate.

# %% [markdown]
# ## Contents
# 1. Imports and fixed execution contract
# 2. Notebook-safe paths, SHA, and fixed assets
# 3. SHA-fixed exp072 / exp209 prediction control
# 4. Raw-well preparation
# 5. Exact exp072 PF with read-only stage diagnostics
# 6. Row ledger and episode attribution
# 7. Kaggle CPU shard orchestration
# 8. Metrics and artifacts

# %%
from __future__ import annotations

import gc
import gzip
import hashlib
import json
import math
import os
import platform
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import yaml
from numba import njit, set_num_threads

EXPERIMENT_NAME = "exp410_likpf_particle_resampling_basin_audit"
OUTPUT_PREFIX = EXPERIMENT_NAME
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
FULL_REPLAY_TRAIN_FEATURES = (
    "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
)
EXP209_CONTROL = (
    "exp209_vs_exp072_exp205_enriched_hmm_exp072_train_features.csv.gz"
)
RADIUS_VALUES = np.asarray([3.0, 5.0, 10.0], dtype=np.float64)
PRIMARY_RADIUS_INDEX = 1


def get_nested(config: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def find_project_root(start: Path = PACKAGE_DIR) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").is_file():
            return candidate
    return start


def config_path() -> Path:
    root = find_project_root()
    candidates = (
        Path.cwd() / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("exp410 config.yaml was not found")


def load_config(path: Path | None = None) -> dict[str, Any]:
    resolved = config_path() if path is None else path
    value = yaml.safe_load(resolved.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{resolved} must contain a YAML mapping")
    return value


def output_dir() -> Path:
    if KAGGLE_WORKING_ROOT.is_dir():
        target = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        target = find_project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    target.mkdir(parents=True, exist_ok=True)
    return target


def metrics_path() -> Path:
    if KAGGLE_WORKING_ROOT.is_dir():
        return KAGGLE_WORKING_ROOT / "metrics.json"
    return find_project_root() / "experiments" / EXPERIMENT_NAME / "metrics.json"


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
    if isinstance(value, Path):
        return str(value)
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(to_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed else open
    with opener(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def logical_prediction_sha(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    ordered = frame.sort_values(["well", "row_idx"], kind="stable")
    for well, group in ordered.groupby("well", sort=False):
        well_bytes = str(well).encode("utf-8")
        digest.update(len(well_bytes).to_bytes(4, "little"))
        digest.update(well_bytes)
        digest.update(
            np.ascontiguousarray(group["row_idx"].to_numpy(np.int64)).tobytes()
        )
        digest.update(
            np.ascontiguousarray(group["likpf_mean"].to_numpy(np.float32)).tobytes()
        )
    return digest.hexdigest()


def dataframe_content_sha(
    frame: pd.DataFrame,
    columns: Iterable[str],
    *,
    sort_by: Iterable[str],
) -> str:
    ordered = frame.sort_values(list(sort_by), kind="stable").reset_index(drop=True)
    payload = ordered[list(columns)].to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def find_existing_path(
    *,
    filename: str,
    explicit_path: str | Path | None = None,
    candidates: Iterable[str | Path] | None = None,
) -> Path | None:
    paths: list[Path] = []
    if explicit_path:
        paths.append(Path(explicit_path))
    paths.extend(Path(value) for value in candidates or [])
    paths.extend(
        (
            Path.cwd() / filename,
            Path.cwd() / "assets" / filename,
            find_project_root() / "experiments" / EXPERIMENT_NAME / "assets" / filename,
        )
    )
    for path in paths:
        if path.is_file() and path.stat().st_size > 0:
            return path
    if KAGGLE_INPUT_ROOT.is_dir():
        matches = sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}"))
        for path in matches:
            if path.is_file() and path.stat().st_size > 0:
                return path
    return None


def require_path(
    *,
    filename: str,
    explicit_path: str | Path | None = None,
    candidates: Iterable[str | Path] | None = None,
) -> Path:
    path = find_existing_path(
        filename=filename,
        explicit_path=explicit_path,
        candidates=candidates,
    )
    if path is None:
        raise FileNotFoundError(f"could not resolve required file: {filename}")
    return path


def validate_execution_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    particles = int(get_nested(config, "model.runtime.particles"))
    seeds = int(get_nested(config, "model.runtime.seed_count"))
    if (particles, seeds) != (500, 128):
        raise RuntimeError(f"PF contract changed: particles/seeds={(particles, seeds)}")
    zero_fields = (
        "execution.lightgbm_configs",
        "execution.trained_folds",
        "execution.boosters",
        "execution.models",
        "execution.hmm_well_runs",
        "execution.beam_well_runs",
        "execution.gpu_runs",
    )
    nonzero = {key: get_nested(config, key) for key in zero_fields if int(get_nested(config, key)) != 0}
    if nonzero:
        raise RuntimeError(f"non-PF execution entered exp410: {nonzero}")
    if int(get_nested(config, "execution.active_pf_variants")) != 1:
        raise RuntimeError("exactly one unchanged PF variant is allowed")
    if not bool(get_nested(config, "execution.kaggle_execution_approved")):
        raise RuntimeError("Kaggle execution is not approved")
    if bool(get_nested(config, "runtime.kaggle.enable_gpu")):
        raise RuntimeError("exp410 must remain CPU-only")
    if bool(get_nested(config, "inference.enabled")):
        raise RuntimeError("inference must remain disabled")
    run_stage = os.environ.get(
        "EXP410_RUN_STAGE", str(get_nested(config, "execution.run_stage"))
    )
    shard_index = int(
        os.environ.get(
            "EXP410_ACTIVE_WELL_SHARD_INDEX",
            str(get_nested(config, "execution.active_well_shard_index")),
        )
    )
    shard_count = int(get_nested(config, "execution.well_shard_count"))
    if run_stage not in {"preflight", "full"}:
        raise RuntimeError(f"unsupported run stage: {run_stage}")
    if not 0 <= shard_index < shard_count:
        raise RuntimeError(f"invalid shard {shard_index}/{shard_count}")
    return {
        "run_stage": run_stage,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "particles": particles,
        "seeds": seeds,
    }


# %% [markdown]
# ## 2. Notebook-safe paths, SHA, and fixed assets

# %%
def load_fixed_assets(config: Mapping[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    well_spec = get_nested(config, "data.target_wells")
    episode_spec = get_nested(config, "data.persistent_episodes")
    manifest_spec = get_nested(config, "data.persistent_asset_manifest")
    well_path = require_path(
        filename=str(well_spec["filename"]),
        explicit_path=well_spec.get("local"),
    )
    episode_path = require_path(
        filename=str(episode_spec["filename"]),
        explicit_path=episode_spec.get("local"),
    )
    manifest_path = require_path(
        filename=str(manifest_spec["filename"]),
        explicit_path=manifest_spec.get("local"),
    )
    actual_well_sha = sha256_path(well_path)
    actual_episode_sha = sha256_path(episode_path)
    if actual_well_sha != str(well_spec["expected_sha256"]):
        raise RuntimeError(f"target-well SHA mismatch: {actual_well_sha}")
    if actual_episode_sha != str(episode_spec["expected_sha256"]):
        raise RuntimeError(f"episode SHA mismatch: {actual_episode_sha}")
    wells = pd.read_csv(well_path, dtype={"well": str})
    episodes = pd.read_csv(episode_path, dtype={"well": str, "episode_id": str})
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "target_wells": int(get_nested(config, "validation.expected_target_wells")),
        "episodes": int(get_nested(config, "validation.expected_episodes")),
        "episode_rows": int(get_nested(config, "validation.expected_episode_rows")),
    }
    actual = {
        "target_wells": len(wells),
        "episodes": len(episodes),
        "episode_rows": int(episodes["rows"].sum()),
    }
    if actual != expected:
        raise RuntimeError(f"fixed asset counts changed: {actual} != {expected}")
    if wells["well"].duplicated().any() or episodes["episode_id"].duplicated().any():
        raise RuntimeError("fixed asset contains duplicate well or episode keys")
    return wells, episodes, {
        "target_wells_path": str(well_path),
        "target_wells_sha256": actual_well_sha,
        "episodes_path": str(episode_path),
        "episodes_sha256": actual_episode_sha,
        "asset_manifest_path": str(manifest_path),
        "asset_manifest_sha256": sha256_path(manifest_path),
        "asset_manifest": manifest,
    }


# %% [markdown]
# ## 3. SHA-fixed exp072 / exp209 prediction control

# %%
def read_filtered_csv(
    path: Path,
    *,
    usecols: list[str],
    target_wells: set[str],
    chunksize: int = 300_000,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        dtype={"id": str, "well": str},
        chunksize=chunksize,
        low_memory=False,
    ):
        selected = chunk.loc[chunk["well"].astype(str).isin(target_wells)]
        if not selected.empty:
            pieces.append(selected.copy())
    if not pieces:
        raise RuntimeError(f"no target wells found in {path}")
    return pd.concat(pieces, ignore_index=True)


def row_indices_from_ids(ids: pd.Series) -> np.ndarray:
    extracted = ids.astype(str).str.extract(r"_(\d+)$", expand=False)
    values = pd.to_numeric(extracted, errors="coerce").to_numpy()
    if np.isnan(values).any():
        raise ValueError("could not recover row_idx from id")
    return values.astype(np.int64)


def load_fixed_prediction_control(
    config: Mapping[str, Any],
    target_wells: set[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    exp072_path = require_path(
        filename=FULL_REPLAY_TRAIN_FEATURES,
        explicit_path=get_nested(config, "data.exp072_train_feature_cache_local"),
        candidates=get_nested(config, "data.exp072_train_feature_cache_candidates"),
    )
    exp209_path = require_path(
        filename=EXP209_CONTROL,
        explicit_path=get_nested(config, "data.exp209_enriched_likpf_control_local"),
        candidates=get_nested(config, "data.exp209_enriched_likpf_control_candidates"),
    )
    source_specs = (
        (
            "exp072",
            exp072_path,
            str(get_nested(config, "data.exp072_train_feature_cache_expected_sha256")),
            str(
                get_nested(
                    config,
                    "data.exp072_train_feature_cache_expected_decompressed_sha256",
                )
            ),
        ),
        (
            "exp209",
            exp209_path,
            str(get_nested(config, "data.exp209_enriched_likpf_control_expected_sha256")),
            str(
                get_nested(
                    config,
                    "data.exp209_enriched_likpf_control_expected_decompressed_sha256",
                )
            ),
        ),
    )
    source_meta: dict[str, Any] = {}
    for name, path, expected_raw, expected_decompressed in source_specs:
        raw_sha = sha256_path(path)
        decompressed_sha = sha256_path(path, decompressed=path.suffix == ".gz")
        if raw_sha != expected_raw or decompressed_sha != expected_decompressed:
            raise RuntimeError(
                f"{name} SHA mismatch: raw={raw_sha}, decompressed={decompressed_sha}"
            )
        source_meta[name] = {
            "path": str(path),
            "sha256": raw_sha,
            "decompressed_sha256": decompressed_sha,
        }

    base_cols = ["id", "well", "target", "last_known_tvt", "md_since"]
    base = read_filtered_csv(
        exp072_path,
        usecols=base_cols,
        target_wells=target_wells,
    )
    control_cols = base_cols + ["hmm_mean_tvt", "hmm_minus_likpf_mean"]
    control = read_filtered_csv(
        exp209_path,
        usecols=control_cols,
        target_wells=target_wells,
    )
    for frame in (base, control):
        frame["id"] = frame["id"].astype(str)
        frame["well"] = frame["well"].astype(str)
        for column in frame.columns:
            if column not in {"id", "well"}:
                frame[column] = pd.to_numeric(frame[column], errors="raise").astype(
                    np.float32
                )
    if base["id"].duplicated().any() or control["id"].duplicated().any():
        raise RuntimeError("duplicate IDs in exp072/exp209 fixed sources")
    control["likpf_mean"] = (
        control["hmm_mean_tvt"] - control["hmm_minus_likpf_mean"]
    ).astype(np.float32)
    merged = base.merge(
        control[base_cols + ["likpf_mean"]],
        on="id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_control"),
        sort=False,
    )
    if merged["likpf_mean"].isna().any():
        raise RuntimeError("exp209 control does not cover exp072 target rows")
    if not (merged["well"] == merged["well_control"]).all():
        raise RuntimeError("exp072/exp209 well mismatch")
    for column in ("target", "last_known_tvt", "md_since"):
        left = merged[column].to_numpy(np.float32)
        right = merged[f"{column}_control"].to_numpy(np.float32)
        if not np.array_equal(left, right):
            raise RuntimeError(f"exp072/exp209 {column} mismatch")
    drop_cols = [
        "well_control",
        "target_control",
        "last_known_tvt_control",
        "md_since_control",
    ]
    merged = merged.drop(columns=drop_cols)
    merged["row_idx"] = row_indices_from_ids(merged["id"]).astype(np.int32)
    merged["true_tvt"] = (
        merged["last_known_tvt"].to_numpy(np.float32)
        + merged["target"].to_numpy(np.float32)
    ).astype(np.float32)
    merged = merged.sort_values(["well", "row_idx"], kind="stable").reset_index(drop=True)
    logical_sha = logical_prediction_sha(merged)
    expected_logical_sha = str(
        get_nested(config, "data.fixed_prediction_subset_content_sha256")
    )
    if logical_sha != expected_logical_sha:
        raise RuntimeError(
            f"fixed prediction content SHA mismatch: {logical_sha} != {expected_logical_sha}"
        )
    if len(merged) != int(get_nested(config, "validation.expected_target_suffix_rows")):
        raise RuntimeError(f"target suffix rows changed: {len(merged)}")
    if merged["well"].nunique() != len(target_wells):
        raise RuntimeError("not all target wells were recovered")
    source_meta["fixed_prediction"] = {
        "rows": int(len(merged)),
        "wells": int(merged["well"].nunique()),
        "logical_sha256": logical_sha,
        "reconstruction": "float32(hmm_mean_tvt - hmm_minus_likpf_mean)",
    }
    return merged, source_meta


# %% [markdown]
# ## 4. Raw-well preparation

# %%
@dataclass(frozen=True)
class PreparedWell:
    well: str
    ids: np.ndarray
    row_idx: np.ndarray
    md_since: np.ndarray
    truth: np.ndarray
    fixed_prediction: np.ndarray
    raw_gr_missing: np.ndarray
    md: np.ndarray
    z: np.ndarray
    gr: np.ndarray
    gr_grid: np.ndarray
    grid_min: float
    grid_step: float
    gr_sigma: float
    last_surface: float
    initial_surface_rate: float
    audit_mask: np.ndarray
    diagnostic_mask: np.ndarray
    clamp_min_tvt: float
    clamp_max_tvt: float


def resolve_train_dir(config: Mapping[str, Any]) -> Path:
    root = find_project_root()
    configured = Path(str(get_nested(config, "data.train_dir")))
    candidates = (
        configured,
        root / configured,
        Path("/kaggle/input/rogii-wellbore-geology-prediction/train"),
    )
    for candidate in candidates:
        if candidate.is_dir() and next(
            candidate.glob("*__horizontal_well.csv"), None
        ):
            return candidate
    if KAGGLE_INPUT_ROOT.is_dir():
        matches = sorted(KAGGLE_INPUT_ROOT.glob("**/*__horizontal_well.csv"))
        if matches:
            parent_counts: dict[Path, int] = {}
            for match in matches:
                parent_counts[match.parent] = parent_counts.get(match.parent, 0) + 1
            return max(parent_counts, key=parent_counts.get)
    raise FileNotFoundError("could not resolve raw train directory")


def exp072_stable_seed(*parts: Any, modulo: int = 2_147_483_647) -> int:
    key = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo + 1


def initial_surface_velocity(prefix: pd.DataFrame) -> float:
    tail = prefix.tail(30)
    tvt = pd.to_numeric(tail["TVT_input"], errors="coerce").to_numpy(np.float64)
    z = pd.to_numeric(tail["Z"], errors="coerce").to_numpy(np.float64)
    md = pd.to_numeric(tail["MD"], errors="coerce").to_numpy(np.float64)
    delta_md = np.diff(md)
    delta_surface = np.diff(tvt) + np.diff(z)
    finite = np.isfinite(delta_md) & np.isfinite(delta_surface) & (delta_md > 0.0)
    if int(finite.sum()) < 3:
        return 0.0
    return float(np.median(delta_surface[finite] / delta_md[finite]))


def make_audit_masks(
    row_idx: np.ndarray,
    well_episodes: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    audit = np.zeros(len(row_idx), dtype=np.uint8)
    if len(audit):
        audit[0] = 1
    for episode in well_episodes.itertuples(index=False):
        selected = (
            (row_idx >= int(episode.audit_start_row_idx))
            & (row_idx < int(episode.end_row_idx_exclusive))
        )
        audit[selected] = 1
    diagnostic = audit.copy()
    starts = np.flatnonzero(
        (audit == 1) & np.concatenate(([True], audit[:-1] == 0))
    )
    for start in starts:
        if start > 0:
            diagnostic[start - 1] = 1
    return audit, diagnostic


def prepare_well(
    *,
    well: str,
    cache_rows: pd.DataFrame,
    well_episodes: pd.DataFrame,
    train_dir: Path,
    config: Mapping[str, Any],
) -> PreparedWell:
    horizontal_path = train_dir / f"{well}__horizontal_well.csv"
    typewell_path = train_dir / f"{well}__typewell.csv"
    if not horizontal_path.is_file() or not typewell_path.is_file():
        raise FileNotFoundError(f"{well}: raw horizontal/typewell input missing")
    horizontal = pd.read_csv(horizontal_path, low_memory=False)
    typewell = pd.read_csv(typewell_path, low_memory=False).sort_values(
        "TVT", kind="stable"
    )
    required_horizontal = {"MD", "Z", "GR", "TVT_input"}
    required_typewell = {"TVT", "GR"}
    if not required_horizontal.issubset(horizontal.columns):
        raise ValueError(f"{well}: horizontal columns missing")
    if not required_typewell.issubset(typewell.columns):
        raise ValueError(f"{well}: typewell columns missing")

    rows = cache_rows.sort_values("row_idx", kind="stable").reset_index(drop=True)
    row_idx = rows["row_idx"].to_numpy(np.int64)
    if len(np.unique(row_idx)) != len(row_idx) or (
        len(row_idx) > 1 and not np.all(np.diff(row_idx) == 1)
    ):
        raise ValueError(f"{well}: evaluation row_idx is not unique and contiguous")
    if int(row_idx[0]) < 0 or int(row_idx[-1]) >= len(horizontal):
        raise ValueError(f"{well}: evaluation row_idx outside horizontal input")

    raw_horizontal_gr = pd.to_numeric(horizontal["GR"], errors="coerce")
    typewell_tvt = pd.to_numeric(typewell["TVT"], errors="coerce").to_numpy(
        np.float64
    )
    typewell_gr_series = pd.to_numeric(typewell["GR"], errors="coerce")
    typewell_gr_mean = float(typewell_gr_series.mean())
    typewell_gr = typewell_gr_series.fillna(typewell_gr_mean).to_numpy(np.float64)
    if (
        not np.isfinite(typewell_tvt).all()
        or not np.isfinite(typewell_gr).all()
        or len(typewell_tvt) < 3
    ):
        raise ValueError(f"{well}: invalid typewell surface")

    masked = horizontal.iloc[: int(row_idx[-1]) + 1].copy()
    masked.loc[row_idx, "TVT_input"] = np.nan
    known = masked.loc[pd.to_numeric(masked["TVT_input"], errors="coerce").notna()]
    minimum_known = int(get_nested(config, "model.runtime.min_known_prefix_rows"))
    if len(known) < minimum_known:
        raise ValueError(f"{well}: known prefix too short: {len(known)}")
    known_tvt = pd.to_numeric(known["TVT_input"], errors="coerce").to_numpy(
        np.float64
    )
    known_gr = (
        pd.to_numeric(known["GR"], errors="coerce")
        .fillna(0.0)
        .to_numpy(np.float64)
    )
    sigma = float(
        np.clip(
            np.nanstd(known_gr - np.interp(known_tvt, typewell_tvt, typewell_gr)),
            float(get_nested(config, "model.runtime.gr_sigma_min")),
            float(get_nested(config, "model.runtime.gr_sigma_max")),
        )
    )
    last_prefix = known.iloc[-1]
    last_surface = float(last_prefix["TVT_input"]) + float(last_prefix["Z"])
    initial_rate = initial_surface_velocity(known)

    grid_step = float(get_nested(config, "model.runtime.grid_step"))
    grid_min = float(np.nanmin(typewell_tvt))
    grid_max = float(np.nanmax(typewell_tvt))
    tvt_grid = np.arange(
        grid_min, grid_max + grid_step, grid_step, dtype=np.float64
    )
    gr_grid = np.interp(tvt_grid, typewell_tvt, typewell_gr).astype(np.float64)
    filled_horizontal_gr = (
        raw_horizontal_gr.interpolate(limit_direction="both")
        .fillna(typewell_gr_mean)
        .to_numpy(np.float64)
    )
    eval_rows = masked.loc[row_idx]
    audit_mask, diagnostic_mask = make_audit_masks(row_idx, well_episodes)
    truth = rows["true_tvt"].to_numpy(np.float32).astype(np.float64)
    fixed_prediction = rows["likpf_mean"].to_numpy(np.float32).astype(np.float64)
    if not np.isfinite(truth).all() or not np.isfinite(fixed_prediction).all():
        raise ValueError(f"{well}: non-finite truth/control")
    return PreparedWell(
        well=well,
        ids=rows["id"].astype(str).to_numpy(),
        row_idx=row_idx,
        md_since=rows["md_since"].to_numpy(np.float32).astype(np.float64),
        truth=truth,
        fixed_prediction=fixed_prediction,
        raw_gr_missing=raw_horizontal_gr.iloc[row_idx].isna().to_numpy(np.uint8),
        md=pd.to_numeric(eval_rows["MD"], errors="raise").to_numpy(np.float64),
        z=pd.to_numeric(eval_rows["Z"], errors="raise").to_numpy(np.float64),
        gr=filled_horizontal_gr[row_idx].astype(np.float64),
        gr_grid=gr_grid,
        grid_min=grid_min,
        grid_step=grid_step,
        gr_sigma=max(sigma, 1.0e-6),
        last_surface=last_surface,
        initial_surface_rate=initial_rate,
        audit_mask=audit_mask,
        diagnostic_mask=diagnostic_mask,
        clamp_min_tvt=grid_min - 100.0,
        clamp_max_tvt=grid_min + len(gr_grid) * grid_step + 100.0,
    )


# %% [markdown]
# ## 5. Exact exp072 PF with read-only stage diagnostics

# %%
@dataclass(frozen=True)
class PfAuditRun:
    seed_predictions: np.ndarray
    log_likelihoods: np.ndarray
    stage_truth_mass: np.ndarray
    stage_candidate_mass: np.ndarray
    stage_mean_tvt: np.ndarray
    stage_mean_rate: np.ndarray
    stage_truth_support_fraction: np.ndarray
    ess_mean: np.ndarray
    resampled_seed_fraction: np.ndarray
    unique_ancestor_fraction: np.ndarray
    max_offspring_fraction: np.ndarray
    transition_escape_seed_fraction: np.ndarray
    emission_escape_seed_fraction: np.ndarray
    resampling_extinction_seed_fraction: np.ndarray
    within_seed_multiplicity_fraction: np.ndarray
    truth_close_seed_fraction: np.ndarray
    candidate_close_seed_fraction: np.ndarray


@njit(cache=True, nogil=True)
def _interp1(grid: np.ndarray, value: float, vmin: float, step: float) -> float:
    index = int((value - vmin) / step)
    if index < 0:
        return grid[0]
    last = len(grid) - 1
    if index >= last:
        return grid[last]
    fraction = (value - vmin) / step - index
    return grid[index] * (1.0 - fraction) + grid[index + 1] * fraction


@njit(cache=True, nogil=True)
def _truth_candidate_log_ratio(
    truth_mass: float,
    candidate_mass: float,
) -> float:
    floor = 1.0e-12
    return np.log(max(truth_mass, floor)) - np.log(max(candidate_mass, floor))


@njit(cache=True, nogil=True)
def _exp072_likpf_particle_audit(
    md_v: np.ndarray,
    z_v: np.ndarray,
    gr_v: np.ndarray,
    gr_grid: np.ndarray,
    vmin: float,
    step: float,
    gr_sigma: float,
    last_surface: float,
    init_rate: float,
    n_particles: int,
    n_seeds: int,
    seed_base: int,
    momentum: float,
    velocity_noise: float,
    position_noise: float,
    resample_pos_noise: float,
    resample_velocity_noise: float,
    resample_threshold: float,
    init_spread: float,
    truth_v: np.ndarray,
    candidate_v: np.ndarray,
    diagnostic_mask: np.ndarray,
    radii: np.ndarray,
    primary_radius_index: int,
    mass_floor: float,
    log_odds_effect: float,
    wrong_seed_radius: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Exact exp072 RNG/state update plus read-only particle diagnostics.

    ``truth_v``, ``candidate_v`` and ``diagnostic_mask`` are read only after a
    particle stage is formed. They never affect transition, likelihood,
    normalization, resampling, roughening, or the persisted estimate.
    """

    n_rows = len(md_v)
    n_radii = len(radii)
    predictions = np.empty((n_seeds, n_rows), dtype=np.float64)
    log_likelihoods = np.empty(n_seeds, dtype=np.float64)
    # stage: 0 predictive, 1 filtered, 2 post-resampling.
    stage_truth_mass = np.zeros((3, n_radii, n_rows), dtype=np.float64)
    stage_candidate_mass = np.zeros((3, n_rows), dtype=np.float64)
    stage_mean_tvt = np.zeros((3, n_rows), dtype=np.float64)
    stage_mean_rate = np.zeros((3, n_rows), dtype=np.float64)
    stage_truth_support = np.zeros((3, n_rows), dtype=np.float64)
    ess_accum = np.zeros(n_rows, dtype=np.float64)
    resampled_accum = np.zeros(n_rows, dtype=np.float64)
    unique_ancestor_accum = np.zeros(n_rows, dtype=np.float64)
    max_offspring_accum = np.zeros(n_rows, dtype=np.float64)
    transition_escape_accum = np.zeros(n_rows, dtype=np.float64)
    emission_escape_accum = np.zeros(n_rows, dtype=np.float64)
    resampling_extinction_accum = np.zeros(n_rows, dtype=np.float64)
    within_seed_multiplicity_accum = np.zeros(n_rows, dtype=np.float64)
    truth_close_seed_accum = np.zeros(n_rows, dtype=np.float64)
    candidate_close_seed_accum = np.zeros(n_rows, dtype=np.float64)
    tmax = vmin + len(gr_grid) * step
    primary_radius = radii[primary_radius_index]

    for seed_index in range(n_seeds):
        # This seed and every RNG call below match exp243's exact exp072 replay.
        np.random.seed(seed_base + seed_index)
        pos = np.empty(n_particles, dtype=np.float64)
        rate = np.empty(n_particles, dtype=np.float64)
        weights = np.empty(n_particles, dtype=np.float64)
        for particle_index in range(n_particles):
            pos[particle_index] = last_surface + init_spread * np.random.randn()
            rate[particle_index] = init_rate + 0.01 * np.random.randn()
            weights[particle_index] = 1.0 / n_particles

        log_likelihood = 0.0
        previous_md = md_v[0] - 1.0
        previous_post_truth_mass = -1.0
        previous_post_candidate_mass = -1.0
        predictive_truth = np.zeros(n_radii, dtype=np.float64)
        filtered_truth = np.zeros(n_radii, dtype=np.float64)
        post_truth = np.zeros(n_radii, dtype=np.float64)

        for row_index in range(n_rows):
            delta_md = md_v[row_index] - previous_md
            if delta_md < 1.0:
                delta_md = 1.0

            # Unchanged exp072 transition and clamp.
            for particle_index in range(n_particles):
                rate[particle_index] = (
                    momentum * rate[particle_index]
                    + velocity_noise * np.random.randn()
                )
                pos[particle_index] += (
                    rate[particle_index] * delta_md
                    + position_noise * np.random.randn()
                )
                tvt_particle = pos[particle_index] - z_v[row_index]
                if tvt_particle < vmin - 100.0:
                    tvt_particle = vmin - 100.0
                if tvt_particle > tmax + 100.0:
                    tvt_particle = tmax + 100.0
                pos[particle_index] = tvt_particle + z_v[row_index]

            diagnostic = diagnostic_mask[row_index] == 1
            for radius_index in range(n_radii):
                predictive_truth[radius_index] = 0.0
            predictive_candidate = 0.0
            predictive_mean_tvt = 0.0
            predictive_mean_rate = 0.0
            predictive_min = 1.0e300
            predictive_max = -1.0e300
            if diagnostic:
                for particle_index in range(n_particles):
                    tvt_particle = pos[particle_index] - z_v[row_index]
                    weight = weights[particle_index]
                    predictive_mean_tvt += weight * tvt_particle
                    predictive_mean_rate += weight * rate[particle_index]
                    if tvt_particle < predictive_min:
                        predictive_min = tvt_particle
                    if tvt_particle > predictive_max:
                        predictive_max = tvt_particle
                    truth_distance = abs(tvt_particle - truth_v[row_index])
                    for radius_index in range(n_radii):
                        if truth_distance <= radii[radius_index]:
                            predictive_truth[radius_index] += weight
                    if abs(tvt_particle - candidate_v[row_index]) <= primary_radius:
                        predictive_candidate += weight

            # Unchanged exp072 raw-GR Gaussian likelihood.
            average_likelihood = 0.0
            for particle_index in range(n_particles):
                expected_gr = _interp1(
                    gr_grid,
                    pos[particle_index] - z_v[row_index],
                    vmin,
                    step,
                )
                residual = (gr_v[row_index] - expected_gr) / gr_sigma
                residual2 = residual * residual
                if residual2 > 600.0:
                    residual2 = 600.0
                likelihood = np.exp(-0.5 * residual2)
                if likelihood < 1.0e-300:
                    likelihood = 1.0e-300
                average_likelihood += weights[particle_index] * likelihood
                weights[particle_index] *= likelihood
            if average_likelihood < 1.0e-300:
                average_likelihood = 1.0e-300
            log_likelihood += np.log(average_likelihood)

            weight_sum = 0.0
            for particle_index in range(n_particles):
                weight_sum += weights[particle_index]
            if weight_sum > 0.0:
                for particle_index in range(n_particles):
                    weights[particle_index] /= weight_sum
            else:
                for particle_index in range(n_particles):
                    weights[particle_index] = 1.0 / n_particles

            for radius_index in range(n_radii):
                filtered_truth[radius_index] = 0.0
            filtered_candidate = 0.0
            filtered_mean_tvt = 0.0
            filtered_mean_rate = 0.0
            if diagnostic:
                for particle_index in range(n_particles):
                    tvt_particle = pos[particle_index] - z_v[row_index]
                    weight = weights[particle_index]
                    filtered_mean_tvt += weight * tvt_particle
                    filtered_mean_rate += weight * rate[particle_index]
                    truth_distance = abs(tvt_particle - truth_v[row_index])
                    for radius_index in range(n_radii):
                        if truth_distance <= radii[radius_index]:
                            filtered_truth[radius_index] += weight
                    if abs(tvt_particle - candidate_v[row_index]) <= primary_radius:
                        filtered_candidate += weight

            inverse_ess = 0.0
            for particle_index in range(n_particles):
                inverse_ess += weights[particle_index] * weights[particle_index]
            ess = 1.0 / inverse_ess
            ess_accum[row_index] += ess

            resampled = False
            unique_fraction = 1.0
            max_offspring_fraction = 1.0 / n_particles
            if ess < resample_threshold * n_particles:
                resampled = True
                cumulative = np.empty(n_particles, dtype=np.float64)
                cumulative_weight = 0.0
                for particle_index in range(n_particles):
                    cumulative_weight += weights[particle_index]
                    cumulative[particle_index] = cumulative_weight
                draw0 = np.random.uniform(0.0, 1.0 / n_particles)
                new_pos = np.empty(n_particles, dtype=np.float64)
                new_rate = np.empty(n_particles, dtype=np.float64)
                cumulative_index = 0
                unique_count = 0
                max_offspring = 0
                current_offspring = 0
                previous_parent = -1
                for particle_index in range(n_particles):
                    draw = draw0 + particle_index / n_particles
                    while (
                        cumulative_index < n_particles - 1
                        and cumulative[cumulative_index] < draw
                    ):
                        cumulative_index += 1
                    if cumulative_index != previous_parent:
                        unique_count += 1
                        current_offspring = 1
                        previous_parent = cumulative_index
                    else:
                        current_offspring += 1
                    if current_offspring > max_offspring:
                        max_offspring = current_offspring
                    new_pos[particle_index] = (
                        pos[cumulative_index]
                        + resample_pos_noise * np.random.randn()
                    )
                    new_rate[particle_index] = (
                        rate[cumulative_index]
                        + resample_velocity_noise * np.random.randn()
                    )
                unique_fraction = unique_count / n_particles
                max_offspring_fraction = max_offspring / n_particles
                for particle_index in range(n_particles):
                    pos[particle_index] = new_pos[particle_index]
                    rate[particle_index] = new_rate[particle_index]
                    weights[particle_index] = 1.0 / n_particles
                resampled_accum[row_index] += 1.0

            for radius_index in range(n_radii):
                post_truth[radius_index] = 0.0
            post_candidate = 0.0
            post_mean_tvt = 0.0
            post_mean_rate = 0.0
            post_min = 1.0e300
            post_max = -1.0e300
            if diagnostic:
                if resampled:
                    for particle_index in range(n_particles):
                        tvt_particle = pos[particle_index] - z_v[row_index]
                        weight = weights[particle_index]
                        post_mean_tvt += weight * tvt_particle
                        post_mean_rate += weight * rate[particle_index]
                        if tvt_particle < post_min:
                            post_min = tvt_particle
                        if tvt_particle > post_max:
                            post_max = tvt_particle
                        truth_distance = abs(tvt_particle - truth_v[row_index])
                        for radius_index in range(n_radii):
                            if truth_distance <= radii[radius_index]:
                                post_truth[radius_index] += weight
                        if abs(tvt_particle - candidate_v[row_index]) <= primary_radius:
                            post_candidate += weight
                else:
                    for radius_index in range(n_radii):
                        post_truth[radius_index] = filtered_truth[radius_index]
                    post_candidate = filtered_candidate
                    post_mean_tvt = filtered_mean_tvt
                    post_mean_rate = filtered_mean_rate
                    post_min = predictive_min
                    post_max = predictive_max

            # Unchanged exp072 persisted estimate.
            estimate = 0.0
            for particle_index in range(n_particles):
                estimate += weights[particle_index] * (
                    pos[particle_index] - z_v[row_index]
                )
            predictions[seed_index, row_index] = estimate

            if diagnostic:
                for radius_index in range(n_radii):
                    stage_truth_mass[0, radius_index, row_index] += (
                        predictive_truth[radius_index]
                    )
                    stage_truth_mass[1, radius_index, row_index] += (
                        filtered_truth[radius_index]
                    )
                    stage_truth_mass[2, radius_index, row_index] += (
                        post_truth[radius_index]
                    )
                stage_candidate_mass[0, row_index] += predictive_candidate
                stage_candidate_mass[1, row_index] += filtered_candidate
                stage_candidate_mass[2, row_index] += post_candidate
                stage_mean_tvt[0, row_index] += predictive_mean_tvt
                stage_mean_tvt[1, row_index] += filtered_mean_tvt
                stage_mean_tvt[2, row_index] += post_mean_tvt
                stage_mean_rate[0, row_index] += predictive_mean_rate
                stage_mean_rate[1, row_index] += filtered_mean_rate
                stage_mean_rate[2, row_index] += post_mean_rate
                truth_value = truth_v[row_index]
                if predictive_min <= truth_value <= predictive_max:
                    stage_truth_support[0, row_index] += 1.0
                    stage_truth_support[1, row_index] += 1.0
                if post_min <= truth_value <= post_max:
                    stage_truth_support[2, row_index] += 1.0
                unique_ancestor_accum[row_index] += unique_fraction
                max_offspring_accum[row_index] += max_offspring_fraction

                primary_predictive_truth = predictive_truth[primary_radius_index]
                primary_filtered_truth = filtered_truth[primary_radius_index]
                primary_post_truth = post_truth[primary_radius_index]
                if previous_post_truth_mass >= 0.0:
                    previous_ratio = _truth_candidate_log_ratio(
                        previous_post_truth_mass,
                        previous_post_candidate_mass,
                    )
                    predictive_ratio = _truth_candidate_log_ratio(
                        primary_predictive_truth,
                        predictive_candidate,
                    )
                    if (
                        (
                            previous_post_truth_mass >= mass_floor
                            and primary_predictive_truth < mass_floor
                        )
                        or predictive_ratio - previous_ratio <= -log_odds_effect
                    ):
                        transition_escape_accum[row_index] += 1.0

                predictive_ratio = _truth_candidate_log_ratio(
                    primary_predictive_truth,
                    predictive_candidate,
                )
                filtered_ratio = _truth_candidate_log_ratio(
                    primary_filtered_truth,
                    filtered_candidate,
                )
                if (
                    (
                        primary_predictive_truth >= mass_floor
                        and primary_filtered_truth < mass_floor
                    )
                    or filtered_ratio - predictive_ratio <= -log_odds_effect
                ):
                    emission_escape_accum[row_index] += 1.0

                post_ratio = _truth_candidate_log_ratio(
                    primary_post_truth,
                    post_candidate,
                )
                if (
                    (
                        primary_filtered_truth >= mass_floor
                        and primary_post_truth < mass_floor
                    )
                    or post_ratio - filtered_ratio <= -log_odds_effect
                ):
                    resampling_extinction_accum[row_index] += 1.0
                if (
                    primary_post_truth >= mass_floor
                    and abs(estimate - truth_value) > wrong_seed_radius
                ):
                    within_seed_multiplicity_accum[row_index] += 1.0
                if abs(estimate - truth_value) <= primary_radius:
                    truth_close_seed_accum[row_index] += 1.0
                if abs(estimate - candidate_v[row_index]) <= primary_radius:
                    candidate_close_seed_accum[row_index] += 1.0
                previous_post_truth_mass = primary_post_truth
                previous_post_candidate_mass = post_candidate

            resampled_accum[row_index] += 0.0  # preserve explicit row accumulator
            previous_md = md_v[row_index]
        log_likelihoods[seed_index] = log_likelihood

    denominator = float(n_seeds)
    return (
        predictions,
        log_likelihoods,
        stage_truth_mass / denominator,
        stage_candidate_mass / denominator,
        stage_mean_tvt / denominator,
        stage_mean_rate / denominator,
        stage_truth_support / denominator,
        ess_accum / denominator,
        resampled_accum / denominator,
        unique_ancestor_accum / denominator,
        max_offspring_accum / denominator,
        transition_escape_accum / denominator,
        emission_escape_accum / denominator,
        resampling_extinction_accum / denominator,
        within_seed_multiplicity_accum / denominator,
        truth_close_seed_accum / denominator,
        candidate_close_seed_accum / denominator,
    )


def run_pf_audit(
    prepared: PreparedWell,
    config: Mapping[str, Any],
) -> tuple[PfAuditRun, dict[str, Any]]:
    runtime = get_nested(config, "model.runtime")
    split = str(get_nested(config, "model.replay.split_key"))
    seed_base = exp072_stable_seed("likpf", split, prepared.well)
    values = _exp072_likpf_particle_audit(
        prepared.md,
        prepared.z,
        prepared.gr,
        prepared.gr_grid,
        prepared.grid_min,
        prepared.grid_step,
        prepared.gr_sigma,
        prepared.last_surface,
        prepared.initial_surface_rate,
        int(runtime["particles"]),
        int(runtime["seed_count"]),
        seed_base,
        float(runtime["momentum"]),
        float(runtime["velocity_noise"]),
        float(runtime["position_noise"]),
        float(runtime["resample_pos_noise"]),
        float(runtime["resample_velocity_noise"]),
        float(runtime["resample_threshold"]),
        float(runtime["init_spread"]),
        prepared.truth,
        prepared.fixed_prediction,
        prepared.diagnostic_mask,
        RADIUS_VALUES,
        PRIMARY_RADIUS_INDEX,
        float(get_nested(config, "audit.mass_floor")),
        float(get_nested(config, "audit.log_odds_effect")),
        float(get_nested(config, "audit.wrong_seed_radius_ft")),
    )
    run = PfAuditRun(*values)
    replay_mean_float64 = run.seed_predictions.mean(axis=0, dtype=np.float64)
    replay_mean = replay_mean_float64.astype(np.float32).astype(np.float64)
    difference = replay_mean - prepared.fixed_prediction
    diagnostics = {
        "well": prepared.well,
        "seed_base": int(seed_base),
        "suffix_rows": int(len(prepared.row_idx)),
        "audit_rows": int(prepared.audit_mask.sum()),
        "diagnostic_rows": int(prepared.diagnostic_mask.sum()),
        "gr_sigma": float(prepared.gr_sigma),
        "initial_surface_rate": float(prepared.initial_surface_rate),
        "parity_max_abs_ft": float(np.max(np.abs(difference))),
        "parity_rmse_ft": float(np.sqrt(np.mean(difference * difference))),
        "replay_rmse_ft": float(
            np.sqrt(np.mean((replay_mean - prepared.truth) ** 2))
        ),
        "fixed_rmse_ft": float(
            np.sqrt(np.mean((prepared.fixed_prediction - prepared.truth) ** 2))
        ),
        "ess_mean": float(np.mean(run.ess_mean)),
        "resampling_rate": float(np.mean(run.resampled_seed_fraction)),
        "raw_gr_missing_fraction": float(np.mean(prepared.raw_gr_missing)),
        "best_log_likelihood_per_row": float(
            np.max(run.log_likelihoods) / len(prepared.row_idx)
        ),
        "log_likelihood_std": float(np.std(run.log_likelihoods)),
    }
    return run, diagnostics


# %% [markdown]
# ## 6. Row ledger and episode attribution

# %%
STAGE_NAMES = ("predictive", "filtered", "postresample")


def safe_log_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    floor = 1.0e-12
    return np.log(np.maximum(numerator, floor)) - np.log(
        np.maximum(denominator, floor)
    )


def finite_rate(values: np.ndarray, md: np.ndarray, initial: float) -> np.ndarray:
    rate = np.empty(len(values), dtype=np.float64)
    previous_value = float(initial)
    previous_md = float(md[0] - 1.0)
    for index in range(len(values)):
        delta_md = max(float(md[index] - previous_md), 1.0)
        rate[index] = (float(values[index]) - previous_value) / delta_md
        previous_value = float(values[index])
        previous_md = float(md[index])
    return rate


def build_row_ledger(
    prepared: PreparedWell,
    run: PfAuditRun,
    config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    replay_mean_float64 = run.seed_predictions.mean(axis=0, dtype=np.float64)
    replay_mean = replay_mean_float64.astype(np.float32).astype(np.float64)
    difference = replay_mean - prepared.fixed_prediction
    parity_atol = float(get_nested(config, "validation.parity_atol_ft"))
    max_abs = float(np.max(np.abs(difference)))
    if max_abs > parity_atol:
        raise RuntimeError(
            f"{prepared.well}: replay parity max_abs={max_abs} > {parity_atol}"
        )

    audit_index = np.flatnonzero(prepared.audit_mask == 1)
    seed_values = run.seed_predictions[:, audit_index]
    truth = prepared.truth[audit_index]
    fixed = prepared.fixed_prediction[audit_index]
    seed_errors = seed_values - truth[None, :]
    best_seed_abs_error = np.min(np.abs(seed_errors), axis=0)
    median_seed_prediction = np.median(seed_values, axis=0)
    seed_prediction_std = np.std(seed_values, axis=0)
    seed_error_mean = np.mean(seed_errors, axis=0)
    seed_error_median = np.median(seed_errors, axis=0)
    positive_seed_fraction = np.mean(seed_errors > 0.0, axis=0)
    mean_seed_squared_error = np.mean(seed_errors * seed_errors, axis=0)
    replay_squared_error = (replay_mean[audit_index] - truth) ** 2

    true_surface = prepared.truth + prepared.z
    fixed_surface = prepared.fixed_prediction + prepared.z
    true_surface_rate = finite_rate(
        true_surface,
        prepared.md,
        prepared.last_surface,
    )
    fixed_surface_rate = finite_rate(
        fixed_surface,
        prepared.md,
        prepared.last_surface,
    )

    ledger = pd.DataFrame(
        {
            "well": prepared.well,
            "id": prepared.ids[audit_index],
            "row_idx": prepared.row_idx[audit_index],
            "suffix_offset": audit_index.astype(np.int32),
            "md_since": prepared.md_since[audit_index],
            "true_tvt": truth,
            "fixed_likpf_mean": fixed,
            "replay_likpf_mean": replay_mean[audit_index],
            "error_ft": fixed - truth,
            "replay_minus_fixed_ft": difference[audit_index],
            "raw_gr_missing": prepared.raw_gr_missing[audit_index].astype(np.uint8),
            "gr_value_used": prepared.gr[audit_index],
            "true_surface_rate": true_surface_rate[audit_index],
            "fixed_surface_rate": fixed_surface_rate[audit_index],
            "truth_outside_clamp": (
                (truth < prepared.clamp_min_tvt)
                | (truth > prepared.clamp_max_tvt)
            ).astype(np.uint8),
            "ess_mean": run.ess_mean[audit_index],
            "resampled_seed_fraction": run.resampled_seed_fraction[audit_index],
            "unique_ancestor_fraction": run.unique_ancestor_fraction[audit_index],
            "max_offspring_fraction": run.max_offspring_fraction[audit_index],
            "transition_escape_seed_fraction": run.transition_escape_seed_fraction[
                audit_index
            ],
            "emission_escape_seed_fraction": run.emission_escape_seed_fraction[
                audit_index
            ],
            "resampling_extinction_seed_fraction": (
                run.resampling_extinction_seed_fraction[audit_index]
            ),
            "within_seed_multiplicity_fraction": (
                run.within_seed_multiplicity_fraction[audit_index]
            ),
            "truth_close_seed_fraction": run.truth_close_seed_fraction[audit_index],
            "candidate_close_seed_fraction": run.candidate_close_seed_fraction[
                audit_index
            ],
            "best_seed_abs_error_ft": best_seed_abs_error,
            "median_seed_prediction": median_seed_prediction,
            "seed_prediction_std_ft": seed_prediction_std,
            "seed_error_mean_ft": seed_error_mean,
            "seed_error_median_ft": seed_error_median,
            "positive_seed_fraction": positive_seed_fraction,
            "mean_seed_squared_error": mean_seed_squared_error,
            "replay_squared_error": replay_squared_error,
            "aggregation_sse_gain_vs_mean_seed": (
                mean_seed_squared_error - replay_squared_error
            ),
            "aggregation_abs_penalty_vs_best_seed_ft": (
                np.abs(replay_mean[audit_index] - truth) - best_seed_abs_error
            ),
        }
    )
    for radius_index, radius in enumerate(RADIUS_VALUES):
        radius_name = f"{int(radius):02d}"
        for stage_index, stage_name in enumerate(STAGE_NAMES):
            ledger[f"{stage_name}_truth_mass_r{radius_name}"] = run.stage_truth_mass[
                stage_index, radius_index, audit_index
            ]
    for stage_index, stage_name in enumerate(STAGE_NAMES):
        ledger[f"{stage_name}_candidate_mass_r05"] = run.stage_candidate_mass[
            stage_index, audit_index
        ]
        ledger[f"{stage_name}_mean_tvt"] = run.stage_mean_tvt[
            stage_index, audit_index
        ]
        ledger[f"{stage_name}_mean_rate"] = run.stage_mean_rate[
            stage_index, audit_index
        ]
        ledger[f"{stage_name}_truth_support_fraction"] = (
            run.stage_truth_support_fraction[stage_index, audit_index]
        )
        ledger[f"{stage_name}_mean_error_ft"] = (
            run.stage_mean_tvt[stage_index, audit_index] - truth
        )
        ledger[f"{stage_name}_rate_error"] = (
            run.stage_mean_rate[stage_index, audit_index]
            - true_surface_rate[audit_index]
        )

    predictive_ratio = safe_log_ratio(
        ledger["predictive_truth_mass_r05"].to_numpy(np.float64),
        ledger["predictive_candidate_mass_r05"].to_numpy(np.float64),
    )
    filtered_ratio = safe_log_ratio(
        ledger["filtered_truth_mass_r05"].to_numpy(np.float64),
        ledger["filtered_candidate_mass_r05"].to_numpy(np.float64),
    )
    post_ratio = safe_log_ratio(
        ledger["postresample_truth_mass_r05"].to_numpy(np.float64),
        ledger["postresample_candidate_mass_r05"].to_numpy(np.float64),
    )
    ledger["predictive_truth_vs_candidate_log_mass_ratio"] = predictive_ratio
    ledger["filtered_truth_vs_candidate_log_mass_ratio"] = filtered_ratio
    ledger["postresample_truth_vs_candidate_log_mass_ratio"] = post_ratio
    ledger["emission_truth_vs_candidate_log_ratio_delta"] = (
        filtered_ratio - predictive_ratio
    )
    ledger["resampling_truth_vs_candidate_log_ratio_delta"] = (
        post_ratio - filtered_ratio
    )
    transition_delta = (
        pd.Series(predictive_ratio) - pd.Series(post_ratio).shift(1)
    ).to_numpy(np.float64)
    contiguous_previous = (
        ledger["suffix_offset"].diff().fillna(0).to_numpy(np.float64) == 1.0
    )
    transition_delta[~contiguous_previous] = np.nan
    ledger["transition_truth_vs_candidate_log_ratio_delta"] = transition_delta
    return ledger, {
        "parity_max_abs_ft": max_abs,
        "parity_rmse_ft": float(np.sqrt(np.mean(difference * difference))),
        "row_ledger_rows": int(len(ledger)),
        "row_ledger_content_sha256": dataframe_content_sha(
            ledger,
            ("well", "row_idx", "fixed_likpf_mean", "replay_likpf_mean"),
            sort_by=("well", "row_idx"),
        ),
    }


def first_effect_row(
    rows: pd.DataFrame,
    *,
    fraction_column: str,
    delta_column: str,
    dominant_fraction: float,
    effect: float,
    require_resampling: bool = False,
) -> float:
    condition = (
        rows[fraction_column].to_numpy(np.float64) >= dominant_fraction
    ) | (rows[delta_column].to_numpy(np.float64) <= -effect)
    if require_resampling:
        condition &= rows["resampled_seed_fraction"].to_numpy(np.float64) > 0.0
    indices = np.flatnonzero(condition)
    return float(rows["row_idx"].iloc[int(indices[0])]) if indices.size else np.nan


def episode_cause(
    *,
    audit_rows: pd.DataFrame,
    episode_rows: pd.DataFrame,
    episode: pd.Series,
    config: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    mass_floor = float(get_nested(config, "audit.mass_floor"))
    effect = float(get_nested(config, "audit.log_odds_effect"))
    dominant = float(get_nested(config, "audit.dominant_row_fraction"))
    radius = float(get_nested(config, "audit.basin_radius_ft"))
    onset_end = min(
        int(episode["end_row_idx_exclusive"]),
        int(episode["start_row_idx"]) + 32,
    )
    onset = audit_rows.loc[
        audit_rows["row_idx"] < onset_end
    ].sort_values("row_idx", kind="stable")
    row0 = audit_rows.loc[audit_rows["suffix_offset"] == 0]
    initial_miss = bool(
        int(episode["start_suffix_offset"]) <= 128
        and not row0.empty
        and float(row0["postresample_truth_mass_r05"].iloc[0]) < mass_floor
        and abs(float(row0["error_ft"].iloc[0])) > radius
    )
    transition_row = first_effect_row(
        onset,
        fraction_column="transition_escape_seed_fraction",
        delta_column="transition_truth_vs_candidate_log_ratio_delta",
        dominant_fraction=dominant,
        effect=effect,
    )
    emission_row = first_effect_row(
        onset,
        fraction_column="emission_escape_seed_fraction",
        delta_column="emission_truth_vs_candidate_log_ratio_delta",
        dominant_fraction=dominant,
        effect=effect,
    )
    resampling_row = first_effect_row(
        onset,
        fraction_column="resampling_extinction_seed_fraction",
        delta_column="resampling_truth_vs_candidate_log_ratio_delta",
        dominant_fraction=dominant,
        effect=effect,
        require_resampling=True,
    )
    events = [
        ("transition_propagation_escape", transition_row, 0),
        ("gr_emission", emission_row, 1),
        ("resampling_particle_extinction", resampling_row, 2),
    ]
    present_events = [item for item in events if np.isfinite(item[1])]
    present_events.sort(key=lambda item: (item[1], item[2]))

    within_fraction = float(
        np.mean(
            episode_rows["within_seed_multiplicity_fraction"].to_numpy(np.float64)
            >= dominant
        )
    )
    across_row = (
        (
            episode_rows["truth_close_seed_fraction"].to_numpy(np.float64)
            >= mass_floor
        )
        & (
            episode_rows["best_seed_abs_error_ft"].to_numpy(np.float64)
            <= radius
        )
        & (np.abs(episode_rows["error_ft"].to_numpy(np.float64)) > 10.0)
    )
    across_fraction = float(np.mean(across_row))
    support_shortage_fraction = float(
        np.mean(
            episode_rows["postresample_truth_support_fraction"].to_numpy(np.float64)
            < dominant
        )
    )
    clamp_fraction = float(
        np.mean(episode_rows["truth_outside_clamp"].to_numpy(np.float64))
    )

    if initial_miss:
        cause = "initial_condition_support_miss"
    elif present_events:
        cause = present_events[0][0]
        if cause == "gr_emission":
            event_row = int(present_events[0][1])
            missing = bool(
                onset.loc[onset["row_idx"] == event_row, "raw_gr_missing"].iloc[0]
            )
            cause = (
                "gr_emission_imputation"
                if missing
                else "gr_emission_alias_observed"
            )
    elif within_fraction >= dominant:
        cause = "within_seed_particle_mean_multiplicity"
    elif across_fraction >= dominant:
        cause = "across_seed_aggregation_multiplicity"
    elif clamp_fraction >= dominant or support_shortage_fraction >= dominant:
        cause = "support_or_clamp_shortage"
    else:
        cause = "mixed_or_unresolved"

    diagnostics = {
        "initial_condition_support_miss": initial_miss,
        "first_transition_effect_row_idx": transition_row,
        "first_emission_effect_row_idx": emission_row,
        "first_resampling_effect_row_idx": resampling_row,
        "within_seed_multiplicity_row_fraction": within_fraction,
        "across_seed_aggregation_row_fraction": across_fraction,
        "support_shortage_row_fraction": support_shortage_fraction,
        "clamp_outside_row_fraction": clamp_fraction,
        "transition_overlap": bool(np.isfinite(transition_row)),
        "emission_overlap": bool(np.isfinite(emission_row)),
        "resampling_overlap": bool(np.isfinite(resampling_row)),
    }
    return cause, diagnostics


def summarize_episodes_for_well(
    *,
    ledger: pd.DataFrame,
    episodes: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    summaries: list[dict[str, Any]] = []
    for _, episode in episodes.sort_values("start_row_idx", kind="stable").iterrows():
        audit_rows = ledger.loc[
            (ledger["row_idx"] >= int(episode["audit_start_row_idx"]))
            & (ledger["row_idx"] < int(episode["end_row_idx_exclusive"]))
        ].copy()
        episode_rows = audit_rows.loc[
            audit_rows["row_idx"] >= int(episode["start_row_idx"])
        ].copy()
        if len(episode_rows) != int(episode["rows"]):
            raise RuntimeError(
                f"{episode['episode_id']}: row coverage {len(episode_rows)}/{episode['rows']}"
            )
        cause, diagnostics = episode_cause(
            audit_rows=audit_rows,
            episode_rows=episode_rows,
            episode=episode,
            config=config,
        )
        errors = episode_rows["error_ft"].to_numpy(np.float64)
        sse = float(np.sum(errors * errors))
        if not np.isclose(sse, float(episode["episode_sse"]), rtol=2.0e-6, atol=0.1):
            raise RuntimeError(
                f"{episode['episode_id']}: episode SSE mismatch {sse}/{episode['episode_sse']}"
            )
        summary: dict[str, Any] = {
            "episode_id": str(episode["episode_id"]),
            "well": str(episode["well"]),
            "shard_index": int(episode["shard_index"]),
            "start_row_idx": int(episode["start_row_idx"]),
            "end_row_idx_exclusive": int(episode["end_row_idx_exclusive"]),
            "rows": int(len(episode_rows)),
            "audit_rows": int(len(audit_rows)),
            "cause": cause,
            "episode_sse": sse,
            "rmse_ft": float(np.sqrt(np.mean(errors * errors))),
            "mean_error_ft": float(np.mean(errors)),
            "error_sign_consistency": float(
                max(np.mean(errors > 0.0), np.mean(errors < 0.0))
            ),
            "raw_gr_missing_fraction": float(episode_rows["raw_gr_missing"].mean()),
            "resampled_seed_fraction_mean": float(
                episode_rows["resampled_seed_fraction"].mean()
            ),
            "ess_mean": float(episode_rows["ess_mean"].mean()),
            "unique_ancestor_fraction_mean": float(
                episode_rows["unique_ancestor_fraction"].mean()
            ),
            "max_offspring_fraction_mean": float(
                episode_rows["max_offspring_fraction"].mean()
            ),
            "predictive_truth_mass_r05_mean": float(
                episode_rows["predictive_truth_mass_r05"].mean()
            ),
            "filtered_truth_mass_r05_mean": float(
                episode_rows["filtered_truth_mass_r05"].mean()
            ),
            "postresample_truth_mass_r05_mean": float(
                episode_rows["postresample_truth_mass_r05"].mean()
            ),
            "truth_close_seed_fraction_mean": float(
                episode_rows["truth_close_seed_fraction"].mean()
            ),
            "best_seed_abs_error_ft_mean": float(
                episode_rows["best_seed_abs_error_ft"].mean()
            ),
            "seed_prediction_std_ft_mean": float(
                episode_rows["seed_prediction_std_ft"].mean()
            ),
            "aggregation_abs_penalty_vs_best_seed_ft_mean": float(
                episode_rows["aggregation_abs_penalty_vs_best_seed_ft"].mean()
            ),
            "transition_escape_seed_fraction_max": float(
                audit_rows["transition_escape_seed_fraction"].max()
            ),
            "emission_escape_seed_fraction_max": float(
                audit_rows["emission_escape_seed_fraction"].max()
            ),
            "resampling_extinction_seed_fraction_max": float(
                audit_rows["resampling_extinction_seed_fraction"].max()
            ),
            **diagnostics,
        }
        summaries.append(summary)
    return pd.DataFrame(summaries)


def cause_summary_frame(episodes: pd.DataFrame) -> pd.DataFrame:
    total_sse = float(episodes["episode_sse"].sum())
    grouped = (
        episodes.groupby("cause", sort=True)
        .agg(
            episodes=("episode_id", "size"),
            wells=("well", "nunique"),
            rows=("rows", "sum"),
            episode_sse=("episode_sse", "sum"),
        )
        .reset_index()
    )
    grouped["episode_fraction"] = grouped["episodes"] / max(len(episodes), 1)
    grouped["sse_fraction"] = grouped["episode_sse"] / max(total_sse, 1.0e-12)
    return grouped.sort_values(
        ["episode_sse", "cause"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)


def threshold_sensitivity_frame(
    *,
    ledgers: pd.DataFrame,
    episodes: pd.DataFrame,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    sensitivity = get_nested(config, "audit.threshold_sensitivity")
    total_sse = float(episodes["episode_sse"].sum())
    keys = [
        (
            float(radius),
            float(mass_floor),
            float(effect),
            float(dominant),
            stage,
        )
        for radius in sensitivity["basin_radius_ft"]
        for mass_floor in sensitivity["mass_floor"]
        for effect in sensitivity["log_odds_effect"]
        for dominant in sensitivity["dominant_row_fraction"]
        for stage in ("transition", "emission", "resampling")
    ]
    accumulator = {
        key: {"episodes": 0, "episode_sse": 0.0}
        for key in keys
    }
    ledger_by_well = {
        str(well): group.sort_values("row_idx", kind="stable")
        for well, group in ledgers.groupby("well", sort=False)
    }
    for episode in episodes.itertuples(index=False):
        well_ledger = ledger_by_well[str(episode.well)]
        scope = well_ledger.loc[
            (
                well_ledger["row_idx"]
                >= int(getattr(episode, "audit_start_row_idx", episode.start_row_idx))
            )
            & (
                well_ledger["row_idx"]
                < min(
                    int(episode.end_row_idx_exclusive),
                    int(episode.start_row_idx) + 32,
                )
            )
        ]
        if scope.empty:
            continue
        resampled = (
            scope["resampled_seed_fraction"].to_numpy(np.float64) > 0.0
        )
        episode_sse = float(episode.episode_sse)
        for radius in sensitivity["basin_radius_ft"]:
            radius_value = float(radius)
            radius_name = f"{int(radius_value):02d}"
            predictive = scope[
                f"predictive_truth_mass_r{radius_name}"
            ].to_numpy(np.float64)
            filtered = scope[
                f"filtered_truth_mass_r{radius_name}"
            ].to_numpy(np.float64)
            post = scope[
                f"postresample_truth_mass_r{radius_name}"
            ].to_numpy(np.float64)
            previous_post = np.concatenate(([np.nan], post[:-1]))
            for mass_floor in sensitivity["mass_floor"]:
                mass_value = float(mass_floor)
                for effect in sensitivity["log_odds_effect"]:
                    effect_value = float(effect)
                    stage_values = {
                        "transition": (
                            (previous_post >= mass_value)
                            & (predictive < mass_value)
                        )
                        | (
                            np.log(np.maximum(predictive, 1.0e-12))
                            - np.log(np.maximum(previous_post, 1.0e-12))
                            <= -effect_value
                        ),
                        "emission": (
                            (predictive >= mass_value)
                            & (filtered < mass_value)
                        )
                        | (
                            np.log(np.maximum(filtered, 1.0e-12))
                            - np.log(np.maximum(predictive, 1.0e-12))
                            <= -effect_value
                        ),
                        "resampling": (
                            (
                                (filtered >= mass_value)
                                & (post < mass_value)
                            )
                            | (
                                np.log(np.maximum(post, 1.0e-12))
                                - np.log(np.maximum(filtered, 1.0e-12))
                                <= -effect_value
                            )
                        )
                        & resampled,
                    }
                    for dominant in sensitivity["dominant_row_fraction"]:
                        dominant_value = float(dominant)
                        for stage, values in stage_values.items():
                            if float(np.mean(values)) >= dominant_value:
                                key = (
                                    radius_value,
                                    mass_value,
                                    effect_value,
                                    dominant_value,
                                    stage,
                                )
                                accumulator[key]["episodes"] += 1
                                accumulator[key]["episode_sse"] += episode_sse
    rows: list[dict[str, Any]] = []
    for key in keys:
        radius, mass_floor, effect, dominant, stage = key
        values = accumulator[key]
        rows.append(
            {
                "basin_radius_ft": radius,
                "mass_floor": mass_floor,
                "log_odds_effect": effect,
                "dominant_row_fraction": dominant,
                "stage": stage,
                "episodes": int(values["episodes"]),
                "episode_fraction": int(values["episodes"]) / max(len(episodes), 1),
                "episode_sse": float(values["episode_sse"]),
                "sse_fraction": float(values["episode_sse"])
                / max(total_sse, 1.0e-12),
            }
        )
    return pd.DataFrame(rows)


# %% [markdown]
# ## 7. Kaggle CPU shard orchestration

# %%
def write_csv(path: Path, frame: pd.DataFrame, *, gzip_output: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if gzip_output:
        frame.to_csv(path, index=False, compression="gzip")
    else:
        frame.to_csv(path, index=False)
    return path


def max_rss_gb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.
    divisor = 1024.0**2 if platform.system() == "Linux" else 1024.0**3
    return value / divisor


def selected_well_rows(
    target_wells: pd.DataFrame,
    execution: Mapping[str, Any],
    config: Mapping[str, Any],
) -> pd.DataFrame:
    if execution["run_stage"] == "preflight":
        count = int(get_nested(config, "execution.preflight_wells_per_shard"))
        selected = (
            target_wells.sort_values(
                ["shard_index", "suffix_rows", "well"],
                ascending=[True, False, True],
                kind="stable",
            )
            .groupby("shard_index", sort=True)
            .head(count)
            .copy()
        )
    else:
        selected = target_wells.loc[
            target_wells["shard_index"].astype(int)
            == int(execution["shard_index"])
        ].copy()
    if selected.empty:
        raise RuntimeError("execution selected no wells")
    return selected.sort_values(
        ["suffix_rows", "well"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)


def main() -> dict[str, Any]:
    started = time.perf_counter()
    config = load_config()
    execution = validate_execution_contract(config)
    set_num_threads(int(get_nested(config, "execution.numba_num_threads")))
    target_wells, fixed_episodes, asset_meta = load_fixed_assets(config)
    selected = selected_well_rows(target_wells, execution, config)
    selected_set = set(selected["well"].astype(str))
    selected_episodes = fixed_episodes.loc[
        fixed_episodes["well"].astype(str).isin(selected_set)
    ].copy()
    selected_episode_rows = int(selected_episodes["rows"].sum())
    selected_suffix_rows = int(selected["suffix_rows"].sum())

    print(
        json.dumps(
            {
                "experiment": EXPERIMENT_NAME,
                "run_stage": execution["run_stage"],
                "active_shard": execution["shard_index"],
                "selected_wells": len(selected),
                "selected_suffix_rows": selected_suffix_rows,
                "selected_episodes": len(selected_episodes),
                "selected_episode_rows": selected_episode_rows,
                "particles": execution["particles"],
                "seeds": execution["seeds"],
                "lightgbm_configs": 0,
                "folds": 0,
                "boosters": 0,
                "gpu": False,
                "inference": False,
                "submission": False,
            },
            indent=2,
            sort_keys=True,
        )
    )

    all_target_set = set(target_wells["well"].astype(str))
    fixed_control, source_meta = load_fixed_prediction_control(
        config,
        all_target_set,
    )
    train_dir = resolve_train_dir(config)
    print(f"raw train directory: {train_dir}")
    print(
        "fixed prediction control:",
        source_meta["fixed_prediction"]["rows"],
        "rows /",
        source_meta["fixed_prediction"]["wells"],
        "wells /",
        source_meta["fixed_prediction"]["logical_sha256"],
    )

    row_ledgers: list[pd.DataFrame] = []
    episode_summaries: list[pd.DataFrame] = []
    well_manifests: list[dict[str, Any]] = []
    parity_failed = 0
    progress_every = int(get_nested(config, "execution.progress_every_wells"))
    hard_runtime = float(get_nested(config, "execution.hard_runtime_seconds"))
    hard_rss = float(get_nested(config, "execution.hard_peak_rss_gb"))

    for well_number, selected_row in enumerate(selected.itertuples(index=False), start=1):
        well = str(selected_row.well)
        well_started = time.perf_counter()
        cache_rows = fixed_control.loc[fixed_control["well"] == well].copy()
        well_episodes = selected_episodes.loc[
            selected_episodes["well"] == well
        ].copy()
        prepared = prepare_well(
            well=well,
            cache_rows=cache_rows,
            well_episodes=well_episodes,
            train_dir=train_dir,
            config=config,
        )
        run, run_diagnostics = run_pf_audit(prepared, config)
        ledger, ledger_meta = build_row_ledger(prepared, run, config)
        episode_frame = summarize_episodes_for_well(
            ledger=ledger,
            episodes=well_episodes,
            config=config,
        )
        row_ledgers.append(ledger)
        episode_summaries.append(episode_frame)
        elapsed_well = time.perf_counter() - well_started
        manifest_row = {
            **run_diagnostics,
            **ledger_meta,
            "shard_index": int(selected_row.shard_index),
            "episodes": int(len(episode_frame)),
            "episode_rows": int(episode_frame["rows"].sum()),
            "elapsed_seconds": float(elapsed_well),
            "peak_rss_gb_after": max_rss_gb(),
            "status": "ok",
        }
        if run_diagnostics["parity_max_abs_ft"] > float(
            get_nested(config, "validation.parity_atol_ft")
        ):
            parity_failed += 1
            manifest_row["status"] = "parity_failed"
        well_manifests.append(manifest_row)
        del run, prepared, cache_rows
        gc.collect()

        total_elapsed = time.perf_counter() - started
        if total_elapsed > hard_runtime:
            raise RuntimeError(f"hard runtime guard exceeded: {total_elapsed:.1f}s")
        if max_rss_gb() > hard_rss:
            raise RuntimeError(f"hard RSS guard exceeded: {max_rss_gb():.3f}GB")
        if well_number % progress_every == 0 or well_number == len(selected):
            print(
                f"[{well_number:03d}/{len(selected):03d}] {well}: "
                f"{elapsed_well:.1f}s, parity={run_diagnostics['parity_max_abs_ft']:.3e}, "
                f"episodes={len(episode_frame)}, peak_rss={max_rss_gb():.3f}GB"
            )

    row_frame = pd.concat(row_ledgers, ignore_index=True).sort_values(
        ["well", "row_idx"], kind="stable"
    )
    episode_frame = pd.concat(episode_summaries, ignore_index=True).sort_values(
        ["well", "start_row_idx"], kind="stable"
    )
    well_frame = pd.DataFrame(well_manifests).sort_values("well", kind="stable")
    if (
        len(well_frame) != len(selected)
        or len(episode_frame) != len(selected_episodes)
        or int(episode_frame["rows"].sum()) != selected_episode_rows
    ):
        raise RuntimeError(
            "strict selected coverage failed: "
            f"wells={len(well_frame)}/{len(selected)}, "
            f"episodes={len(episode_frame)}/{len(selected_episodes)}, "
            f"rows={int(episode_frame['rows'].sum())}/{selected_episode_rows}"
        )
    if parity_failed:
        raise RuntimeError(f"replay parity failed for {parity_failed} wells")

    cause_frame = cause_summary_frame(episode_frame)
    sensitivity_frame = threshold_sensitivity_frame(
        ledgers=row_frame,
        episodes=selected_episodes,
        config=config,
    )
    output = output_dir()
    row_path = write_csv(
        output / f"{OUTPUT_PREFIX}_row_ledger.csv.gz",
        row_frame,
        gzip_output=True,
    )
    episode_path = write_csv(
        output / f"{OUTPUT_PREFIX}_episode_summary.csv",
        episode_frame,
    )
    cause_path = write_csv(
        output / f"{OUTPUT_PREFIX}_cause_summary.csv",
        cause_frame,
    )
    sensitivity_path = write_csv(
        output / f"{OUTPUT_PREFIX}_threshold_sensitivity.csv",
        sensitivity_frame,
    )
    well_path = write_csv(
        output / f"{OUTPUT_PREFIX}_well_manifest.csv",
        well_frame,
    )

    input_manifest = {
        "experiment": EXPERIMENT_NAME,
        "run_stage": execution["run_stage"],
        "active_shard": execution["shard_index"],
        "train_dir": str(train_dir),
        "fixed_assets": asset_meta,
        "sources": source_meta,
        "selected_wells_content_sha256": dataframe_content_sha(
            selected,
            ("well", "shard_index", "suffix_rows", "episodes", "episode_rows"),
            sort_by=("well",),
        ),
        "selected_episodes_content_sha256": dataframe_content_sha(
            selected_episodes,
            (
                "episode_id",
                "well",
                "start_row_idx",
                "end_row_idx_exclusive",
                "rows",
                "episode_sse",
            ),
            sort_by=("well", "start_row_idx"),
        ),
        "execution": execution,
    }
    input_manifest_path = output / f"{OUTPUT_PREFIX}_input_manifest.json"
    write_json(input_manifest_path, input_manifest)

    elapsed = time.perf_counter() - started
    primary_sse = (
        {
            str(row.cause): float(row.sse_fraction)
            for row in cause_frame.itertuples(index=False)
        }
        if not cause_frame.empty
        else {}
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "complete",
        "run_stage": execution["run_stage"],
        "active_shard": execution["shard_index"],
        "counts": {
            "wells": int(len(well_frame)),
            "suffix_rows": selected_suffix_rows,
            "episodes": int(len(episode_frame)),
            "episode_rows": int(episode_frame["rows"].sum()),
            "row_ledger_rows": int(len(row_frame)),
        },
        "parity": {
            "failed_wells": int(parity_failed),
            "max_abs_ft": float(well_frame["parity_max_abs_ft"].max()),
            "mean_rmse_ft": float(well_frame["parity_rmse_ft"].mean()),
            "atol_ft": float(get_nested(config, "validation.parity_atol_ft")),
        },
        "runtime": {
            "elapsed_seconds": elapsed,
            "peak_rss_gb": max_rss_gb(),
            "mean_well_seconds": float(well_frame["elapsed_seconds"].mean()),
            "projected_full_seconds": float(
                well_frame["elapsed_seconds"].sum()
                * int(get_nested(config, "validation.expected_target_suffix_rows"))
                / max(selected_suffix_rows, 1)
            ),
        },
        "cause_sse_fraction": primary_sse,
        "artifacts": {
            "row_ledger": str(row_path),
            "episode_summary": str(episode_path),
            "cause_summary": str(cause_path),
            "threshold_sensitivity": str(sensitivity_path),
            "well_manifest": str(well_path),
            "input_manifest": str(input_manifest_path),
        },
        "artifact_sha256": {
            "row_ledger_raw": sha256_path(row_path),
            "row_ledger_decompressed": sha256_path(row_path, decompressed=True),
            "episode_summary": sha256_path(episode_path),
            "cause_summary": sha256_path(cause_path),
            "threshold_sensitivity": sha256_path(sensitivity_path),
            "well_manifest": sha256_path(well_path),
            "input_manifest": sha256_path(input_manifest_path),
        },
        "guards": {
            "fixed_prediction_content_sha": (
                source_meta["fixed_prediction"]["logical_sha256"]
                == str(get_nested(config, "data.fixed_prediction_subset_content_sha256"))
            ),
            "strict_selected_coverage": True,
            "all_well_parity": parity_failed == 0,
            "truth_not_used_by_pf_dynamics": True,
            "rng_calls_added": 0,
            "prediction_candidate_created": False,
            "inference": False,
            "submission": False,
        },
    }
    summary_path = output / f"{OUTPUT_PREFIX}_summary.json"
    write_json(summary_path, summary)
    summary["artifact_sha256"]["summary"] = sha256_path(summary_path)

    metrics = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": "complete",
        "metric": "mechanism_attribution_not_candidate_cv",
        "run_stage": execution["run_stage"],
        "active_shard": execution["shard_index"],
        "counts": summary["counts"],
        "parity": summary["parity"],
        "runtime": summary["runtime"],
        "cause_sse_fraction": primary_sse,
        "guards": summary["guards"],
    }
    write_json(metrics_path(), metrics)

    print("Cause summary")
    print(cause_frame.to_string(index=False))
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 8. Metrics and artifacts
#
# The cell above performs one approved CPU-only diagnostic replay per selected
# well and writes no model, inference output, or submission.

# %%
RESULT = (
    None
    if os.environ.get("EXP410_IMPORT_ONLY", "0") == "1"
    else main()
)
