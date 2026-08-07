from __future__ import annotations

from copy import deepcopy
import gzip
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numba import njit
from settings import KAGGLE_INPUT_ROOT, ExperimentPaths, get_nested, load_config

OUTPUT_PREFIX = "exp233_adaptive_outlier_mixture_likelihood_pf"
EXPERIMENT_NAME = "exp233_adaptive_outlier_mixture_likelihood_pf"
FULL_REPLAY_TRAIN_FEATURES = (
    "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz"
)
FULL_REPLAY_FEATURE_SCHEMA = "exp063_full_replay_feature_cache_feature_schema.csv"
EXP209_ENRICHED_LIKPF_CONTROL = (
    "exp209_vs_exp072_exp205_enriched_hmm_exp072_train_features.csv.gz"
)


@dataclass(frozen=True)
class GrFilterSpec:
    name: str
    kind: str
    params: dict[str, Any]
    transition: str
    transition_params: dict[str, Any]


@dataclass(frozen=True)
class OutlierMixtureSpec:
    name: str
    epsilon: float
    component: str
    uniform_gr_min: float
    uniform_gr_max: float


@dataclass(frozen=True)
class PrefixHoldout:
    well: str
    masked: pd.DataFrame
    typewell: pd.DataFrame
    eval_index: np.ndarray
    eval_ids: np.ndarray
    true_tvt: np.ndarray
    target_delta: np.ndarray
    last_known_tvt: float
    last_known_md: float
    cache_md_since: np.ndarray | None
    reference_candidates: pd.DataFrame
    status: dict[str, Any]


@dataclass(frozen=True)
class PfRun:
    preds: np.ndarray
    log_likelihoods: np.ndarray
    ess_mean_by_row: np.ndarray
    resampled_by_row: np.ndarray
    gate_rate_by_row: np.ndarray
    mixture_rate_by_row: np.ndarray
    innovation_by_row: np.ndarray
    change_point_by_row: np.ndarray
    novelty_by_row: np.ndarray
    preupdate_ess_ratio_by_row: np.ndarray
    preupdate_max_weight_by_row: np.ndarray
    interval_p05_by_seed: np.ndarray
    interval_p95_by_seed: np.ndarray
    sigma: float
    filter_metadata: dict[str, Any]


@dataclass(frozen=True)
class StructuralPrior:
    active: bool
    expected_tvt: np.ndarray
    sigma: float
    weight: float
    metadata: dict[str, Any]


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
        return float(value) if np.isfinite(float(value)) else None
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed else Path.open
    with opener(path, "rb") as fp:  # type: ignore[arg-type]
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    key = "::".join(str(part) for part in parts).encode()
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def find_existing_path(
    *,
    filename: str,
    explicit_path: str | Path | None = None,
    candidates: list[str | Path] | tuple[str | Path, ...] | None = None,
) -> Path | None:
    paths: list[Path] = []
    if explicit_path is not None:
        paths.append(Path(explicit_path))
    if candidates:
        paths.extend(Path(item) for item in candidates)
    paths.extend(
        [
            Path.cwd() / filename,
            Path.cwd() / "artifacts" / filename,
            Path("artifacts") / filename,
            Path("experiments")
            / "exp072_exp063_full_replay_feature_cache"
            / "artifacts"
            / filename,
        ]
    )
    for path in paths:
        if path.exists() and path.stat().st_size > 0:
            return path
    if KAGGLE_INPUT_ROOT.exists():
        for path in KAGGLE_INPUT_ROOT.glob(f"**/{filename}"):
            if path.exists() and path.stat().st_size > 0:
                return path
    return None


def require_path(
    *,
    filename: str,
    explicit_path: str | Path | None = None,
    candidates: list[str | Path] | tuple[str | Path, ...] | None = None,
) -> Path:
    path = find_existing_path(filename=filename, explicit_path=explicit_path, candidates=candidates)
    if path is None:
        checked = [str(explicit_path)] if explicit_path is not None else []
        checked.extend(str(item) for item in candidates or [])
        raise FileNotFoundError(f"artifact not found or empty: {filename}. Checked: {checked}")
    return path


def numeric_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise ValueError(f"required column is missing: {column}")
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float32)


def row_indices_from_ids(ids: pd.Series) -> np.ndarray:
    extracted = ids.astype(str).str.extract(r"_(\d+)$", expand=False)
    values = pd.to_numeric(extracted, errors="coerce").to_numpy()
    if np.isnan(values).any():
        bad = ids[pd.isna(extracted)].head(5).tolist()
        raise ValueError(f"Could not recover row index from ids, examples={bad}")
    return values.astype(np.int64)


def reference_candidate_columns(config: dict[str, Any], header: list[str]) -> list[str]:
    requested = [
        str(value) for value in get_nested(config, "data.exp072_reference_candidates") or []
    ]
    return [column for column in requested if column in header]


def read_exp209_reconstructed_likpf_control(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load exp209's enriched cache and reconstruct its retained exp072 likelihood PF mean.

    The released exp072 ML surface deliberately excludes row-level ``likpf_mean``.
    This comparison-only source reconstructs it as
    ``hmm_mean_tvt - hmm_minus_likpf_mean``; it never enters the PF gate or update.
    """

    source = require_path(
        filename=EXP209_ENRICHED_LIKPF_CONTROL,
        explicit_path=get_nested(config, "data.exp209_enriched_likpf_control_local"),
        candidates=get_nested(config, "data.exp209_enriched_likpf_control_candidates"),
    )
    required = [
        "id",
        "well",
        "target",
        "last_known_tvt",
        "md_since",
        "hmm_mean_tvt",
        "hmm_minus_likpf_mean",
    ]
    header = pd.read_csv(source, nrows=0).columns.tolist()
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError(f"{source} is missing exp209 control columns: {missing}")
    control = pd.read_csv(
        source,
        usecols=required,
        dtype={"id": str, "well": str},
        low_memory=False,
    )
    control["id"] = control["id"].astype(str)
    control["well"] = control["well"].astype(str)
    if control["id"].duplicated().any():
        examples = control.loc[control["id"].duplicated(), "id"].head(5).tolist()
        raise ValueError(f"{source} has duplicate control ids, examples={examples}")
    for column in required:
        if column not in {"id", "well"}:
            control[column] = pd.to_numeric(control[column], errors="coerce").astype(np.float32)
    control["likpf_mean"] = (
        control["hmm_mean_tvt"] - control["hmm_minus_likpf_mean"]
    ).astype(np.float32)
    if not np.isfinite(control["likpf_mean"].to_numpy(np.float32)).all():
        invalid = int((~np.isfinite(control["likpf_mean"].to_numpy(np.float32))).sum())
        raise ValueError(f"{source} reconstructed non-finite likpf_mean rows={invalid}")
    return control, {
        "source": str(source),
        "source_sha256": sha256_path(source),
        "source_decompressed_sha256": (
            sha256_path(source, decompressed=True) if source.suffix == ".gz" else None
        ),
        "rows": int(len(control)),
        "wells": int(control["well"].nunique()),
        "reconstruction": "likpf_mean = hmm_mean_tvt - hmm_minus_likpf_mean",
        "source_experiment": "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation",
    }


def align_exp209_likpf_control(
    frame: pd.DataFrame,
    control: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach the reconstructed T=1 control after strict row identity checks."""

    if frame["id"].duplicated().any():
        examples = frame.loc[frame["id"].duplicated(), "id"].head(5).tolist()
        raise ValueError(f"exp072 evaluation cache has duplicate ids, examples={examples}")
    control_columns = ["id", "well", "target", "last_known_tvt", "md_since", "likpf_mean"]
    merged = frame.merge(
        control[control_columns],
        on="id",
        how="left",
        validate="one_to_one",
        sort=False,
        suffixes=("", "_exp209"),
    )
    missing_control = merged["likpf_mean"].isna()
    if missing_control.any():
        examples = merged.loc[missing_control, "id"].head(5).tolist()
        raise ValueError(
            "exp209 reconstructed control is missing ids from the exp072 evaluation cache, "
            f"rows={int(missing_control.sum())}, examples={examples}"
        )
    mismatch_counts: dict[str, int] = {}
    if not (merged["well"] == merged["well_exp209"]).all():
        mismatch_counts["well"] = int((merged["well"] != merged["well_exp209"]).sum())
    for column in ["target", "last_known_tvt", "md_since"]:
        if column not in frame.columns:
            continue
        left = pd.to_numeric(merged[column], errors="coerce").to_numpy(np.float64)
        right = pd.to_numeric(merged[f"{column}_exp209"], errors="coerce").to_numpy(np.float64)
        matches = np.isclose(left, right, rtol=1.0e-6, atol=1.0e-4, equal_nan=True)
        if not matches.all():
            mismatch_counts[column] = int((~matches).sum())
    if mismatch_counts:
        raise ValueError(
            "exp209 reconstructed control does not align with the exp072 evaluation cache: "
            f"{mismatch_counts}"
        )
    source_columns = ["well_exp209", "target_exp209", "last_known_tvt_exp209", "md_since_exp209"]
    return merged.drop(columns=[column for column in source_columns if column in merged.columns]), {
        "alignment": "one_to_one_id_with_well_target_last_known_tvt_md_since_checks",
        "rows": int(len(merged)),
        "mismatch_counts": mismatch_counts,
    }


def read_exp072_eval_cache(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = require_path(
        filename=FULL_REPLAY_TRAIN_FEATURES,
        explicit_path=get_nested(config, "data.exp072_train_feature_cache_local"),
    )
    required = ["id", "well", "target", "last_known_tvt"]
    optional = ["md_since", "eval_len", "known_len"]
    header = pd.read_csv(source, nrows=0).columns.tolist()
    missing = [column for column in required if column not in header]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")
    references = reference_candidate_columns(config, header)
    usecols = required + [column for column in optional if column in header] + references
    max_rows = get_nested(config, "audit.max_rows")
    frame = pd.read_csv(
        source,
        usecols=usecols,
        nrows=None if max_rows is None else int(max_rows),
        dtype={"id": str, "well": str},
        low_memory=False,
    )
    frame["id"] = frame["id"].astype(str)
    frame["well"] = frame["well"].astype(str)
    for column in frame.columns:
        if column not in {"id", "well"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32)
    frame["row_idx"] = row_indices_from_ids(frame["id"]).astype(np.int32)
    frame["true_tvt"] = (
        numeric_array(frame, "last_known_tvt") + numeric_array(frame, "target")
    ).astype(np.float32)
    exp209_control, control_metadata = read_exp209_reconstructed_likpf_control(config)
    frame, alignment_metadata = align_exp209_likpf_control(frame, exp209_control)
    requested_references = [
        str(value) for value in get_nested(config, "data.exp072_reference_candidates") or []
    ]
    references = [column for column in requested_references if column in frame.columns]

    schema_path = find_existing_path(
        filename=FULL_REPLAY_FEATURE_SCHEMA,
        explicit_path=get_nested(config, "data.exp072_feature_schema_local"),
    )
    metadata = {
        "source": str(source),
        "source_sha256": sha256_path(source),
        "source_decompressed_sha256": (
            sha256_path(source, decompressed=True) if source.suffix == ".gz" else None
        ),
        "schema": str(schema_path) if schema_path else None,
        "schema_sha256": sha256_path(schema_path) if schema_path else None,
        "rows": int(len(frame)),
        "wells": int(frame["well"].nunique()),
        "columns": list(frame.columns),
        "reference_candidates_present": references,
        "exp209_reconstructed_likpf_control": {
            **control_metadata,
            **alignment_metadata,
        },
        "max_rows": None if max_rows is None else int(max_rows),
    }
    return frame, metadata


def fill_numeric(values: pd.Series | np.ndarray, fallback: float) -> np.ndarray:
    series = pd.Series(values, dtype="float64")
    if series.notna().any():
        fallback = float(series.mean())
    filled = series.interpolate(limit_direction="both").ffill().bfill().fillna(fallback)
    return filled.to_numpy(np.float32)


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values, dtype="float64")
        .rolling(int(window), center=True, min_periods=1)
        .median()
        .to_numpy(np.float32)
    )


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values, dtype="float64")
        .rolling(int(window), center=True, min_periods=1)
        .mean()
        .to_numpy(np.float32)
    )


def savgol_or_rolling_mean(
    values: np.ndarray,
    *,
    window: int,
    polyorder: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    window = int(window)
    if window % 2 == 0:
        window += 1
    if len(values) < window or window <= int(polyorder):
        effective = max(3, min(len(values), window))
        return rolling_mean(values, effective), {
            "effective_kind": "rolling_mean_short_series",
            "window": int(effective),
        }
    try:
        from scipy.signal import savgol_filter

        return (
            savgol_filter(
                values.astype(np.float64),
                window_length=window,
                polyorder=int(polyorder),
                mode="interp",
            ).astype(np.float32),
            {"effective_kind": "savgol", "window": window, "polyorder": int(polyorder)},
        )
    except Exception as exc:  # pragma: no cover - depends on Kaggle image packages.
        return rolling_mean(values, window), {
            "effective_kind": "rolling_mean_fallback",
            "window": window,
            "polyorder": int(polyorder),
            "fallback_reason": type(exc).__name__,
        }


def parse_filter_specs(config: dict[str, Any]) -> list[GrFilterSpec]:
    specs: list[GrFilterSpec] = []
    raw_specs = get_nested(config, "model.observation_variants")
    if raw_specs is None:
        raw_specs = get_nested(config, "model.gr_filters") or []
    for raw in raw_specs:
        item = dict(raw)
        name = str(item.pop("name"))
        kind = str(item.pop("kind", "raw"))
        transition = str(item.pop("transition", "classic"))
        transition_params = item.pop("transition_params", {})
        if transition_params is None:
            transition_params = {}
        if not isinstance(transition_params, dict):
            raise ValueError(f"transition_params for {name} must be a mapping")
        specs.append(
            GrFilterSpec(
                name=name,
                kind=kind,
                params=item,
                transition=transition,
                transition_params=dict(transition_params),
            )
        )
    if not specs:
        raise ValueError("model.observation_variants must define at least one variant")
    if specs[0].name != "raw" or specs[0].kind != "raw" or specs[0].transition != "classic":
        raise ValueError("first model.observation_variants entry must be raw classic baseline")
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate GR filter names: {names}")
    return specs


def apply_gr_filter(values: np.ndarray, spec: GrFilterSpec) -> tuple[np.ndarray, dict[str, Any]]:
    fallback = float(np.nanmean(values)) if np.isfinite(values).any() else 0.0
    filled = fill_numeric(values, fallback)
    if spec.kind == "raw":
        return filled.astype(np.float32), {"effective_kind": "raw"}
    if spec.kind == "rolling_median":
        window = int(spec.params.get("window", 11))
        return rolling_median(filled, window), {
            "effective_kind": "rolling_median",
            "window": window,
        }
    if spec.kind == "rolling_mean":
        window = int(spec.params.get("window", 21))
        return rolling_mean(filled, window), {
            "effective_kind": "rolling_mean",
            "window": window,
        }
    if spec.kind == "savgol":
        return savgol_or_rolling_mean(
            filled,
            window=int(spec.params.get("window", 31)),
            polyorder=int(spec.params.get("polyorder", 2)),
        )
    raise ValueError(f"unknown GR filter kind: {spec.kind}")


def select_target_wells(
    validation_frame: pd.DataFrame,
    train_dir: Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    surface = get_nested(config, "model.validation_surface") or {}
    include = [str(value) for value in surface.get("well_include", []) if value]
    min_eval = int(surface.get("min_eval_rows", 64))
    max_target_wells = surface.get("max_target_wells")
    summary = (
        validation_frame.groupby("well", sort=False)
        .agg(
            eval_rows=("id", "size"),
            max_md_since=("md_since", "max")
            if "md_since" in validation_frame.columns
            else ("row_idx", "max"),
            mean_md_since=("md_since", "mean")
            if "md_since" in validation_frame.columns
            else ("row_idx", "mean"),
            known_len=("known_len", "max")
            if "known_len" in validation_frame.columns
            else ("row_idx", "min"),
            eval_len=("eval_len", "max")
            if "eval_len" in validation_frame.columns
            else ("id", "size"),
        )
        .reset_index()
    )
    summary["has_horizontal"] = summary["well"].map(
        lambda well: (train_dir / f"{well}__horizontal_well.csv").exists()
    )
    summary["has_typewell"] = summary["well"].map(
        lambda well: (train_dir / f"{well}__typewell.csv").exists()
    )
    summary = summary[
        (summary["eval_rows"] >= min_eval) & summary["has_horizontal"] & summary["has_typewell"]
    ].copy()
    if include:
        selected = summary[summary["well"].isin(include)].copy()
    else:
        mode = str(surface.get("selection_mode", "long_md_since"))
        if mode == "eval_rows":
            selected = summary.sort_values(
                ["eval_rows", "max_md_since", "well"],
                ascending=[False, False, True],
            )
        else:
            selected = summary.sort_values(
                ["max_md_since", "eval_rows", "well"],
                ascending=[False, False, True],
            )
        if max_target_wells is not None:
            selected = selected.head(int(max_target_wells))
    return selected.reset_index(drop=True)


def build_eval_zone_for_well(
    well: str,
    eval_cache_rows: pd.DataFrame,
    train_dir: Path,
    config: dict[str, Any],
) -> PrefixHoldout | None:
    validation_cfg = get_nested(config, "model.validation_surface") or {}
    min_known = int(validation_cfg.get("min_known_prefix_rows", 160))
    min_eval = int(validation_cfg.get("min_eval_rows", 64))

    hw_path = train_dir / f"{well}__horizontal_well.csv"
    tw_path = train_dir / f"{well}__typewell.csv"
    if not hw_path.exists() or not tw_path.exists() or eval_cache_rows.empty:
        return None
    horizontal = pd.read_csv(hw_path, low_memory=False)
    typewell = pd.read_csv(tw_path, low_memory=False).sort_values("TVT").reset_index(drop=True)
    if len(typewell) < 3:
        return None

    cache_rows = eval_cache_rows.copy()
    if "row_idx" not in cache_rows.columns:
        cache_rows["row_idx"] = row_indices_from_ids(cache_rows["id"]).astype(np.int32)
    cache_rows = cache_rows.sort_values("row_idx").reset_index(drop=True)
    eval_index = pd.to_numeric(cache_rows["row_idx"], errors="coerce").to_numpy(np.int64)
    valid_index = (eval_index >= 0) & (eval_index < len(horizontal))
    if not valid_index.all():
        cache_rows = cache_rows.loc[valid_index].reset_index(drop=True)
        eval_index = eval_index[valid_index]
    if len(eval_index) < min_eval:
        return None

    known_mask = pd.to_numeric(horizontal["TVT_input"], errors="coerce").notna().to_numpy()
    known_idx = np.flatnonzero(known_mask)
    if len(known_idx) < min_known:
        return None
    prefix_end = int(eval_index[0])
    known_before = known_idx[known_idx < prefix_end]
    if len(known_before) < min_known:
        return None
    last_known_idx = int(known_before[-1])

    masked = horizontal.iloc[: int(eval_index[-1]) + 1].copy()
    masked.loc[eval_index, "TVT_input"] = np.nan

    target_delta = numeric_array(cache_rows, "target")
    last_known_values = numeric_array(cache_rows, "last_known_tvt")
    last_known_tvt = float(last_known_values[0])
    truth = (last_known_values + target_delta).astype(np.float32)
    last_known_md = float(horizontal.loc[last_known_idx, "MD"])
    md_since = numeric_array(cache_rows, "md_since") if "md_since" in cache_rows.columns else None
    references = reference_candidate_columns(config, cache_rows.columns.tolist())
    status = {
        "well": well,
        "status": "ok",
        "validation_surface": "exp072_TVT_input_missing_equivalent_rows",
        "known_rows": int(len(known_idx)),
        "eval_rows": int(len(eval_index)),
        "last_known_idx": int(last_known_idx),
        "last_known_tvt": last_known_tvt,
        "last_known_md": last_known_md,
        "max_md_since": float(np.nanmax(md_since)) if md_since is not None else None,
        "reference_candidates_present": references,
        "raw_eval_tvt_input_missing_rate": float(
            pd.to_numeric(horizontal.loc[eval_index, "TVT_input"], errors="coerce")
            .isna()
            .mean()
        ),
    }
    return PrefixHoldout(
        well=well,
        masked=masked,
        typewell=typewell,
        eval_index=eval_index,
        eval_ids=cache_rows["id"].astype(str).to_numpy(),
        true_tvt=truth,
        target_delta=target_delta,
        last_known_tvt=last_known_tvt,
        last_known_md=last_known_md,
        cache_md_since=md_since,
        reference_candidates=cache_rows[references].copy() if references else pd.DataFrame(),
        status=status,
    )


def gr_sigma(
    *,
    horizontal: pd.DataFrame,
    filtered_horizontal_gr: np.ndarray,
    eval_start: int,
    tw_tvt: np.ndarray,
    filtered_typewell_gr: np.ndarray,
    config: dict[str, Any],
) -> float:
    runtime = get_nested(config, "model.runtime") or {}
    prefix = horizontal.iloc[:eval_start]
    finite = prefix["TVT_input"].notna() & prefix["GR"].notna()
    if int(finite.sum()) < 20:
        return float(runtime.get("gr_sigma_default", 30.0))
    prefix_pos = prefix.index.to_numpy(np.int64)
    tvt = pd.to_numeric(prefix.loc[finite, "TVT_input"], errors="coerce").to_numpy(np.float64)
    gr = filtered_horizontal_gr[prefix_pos[finite.to_numpy()]].astype(np.float64)
    residual = gr - np.interp(tvt, tw_tvt, filtered_typewell_gr)
    return float(
        np.clip(
            np.nanstd(residual),
            float(runtime.get("gr_sigma_min", 10.0)),
            float(runtime.get("gr_sigma_max", 60.0)),
        )
    )


def initial_velocity(prefix: pd.DataFrame) -> float:
    tail = prefix.tail(30)
    tvt = pd.to_numeric(tail["TVT_input"], errors="coerce").to_numpy(np.float64)
    md = pd.to_numeric(tail["MD"], errors="coerce").to_numpy(np.float64)
    dm = np.diff(md)
    dt = np.diff(tvt)
    finite = np.isfinite(dm) & np.isfinite(dt) & (dm > 0.0)
    if int(finite.sum()) < 3:
        return 0.0
    return float(np.median(dt[finite] / dm[finite]))


def initial_surface_velocity(prefix: pd.DataFrame) -> float:
    tail = prefix.tail(30)
    tvt = pd.to_numeric(tail["TVT_input"], errors="coerce").to_numpy(np.float64)
    z = pd.to_numeric(tail["Z"], errors="coerce").to_numpy(np.float64)
    md = pd.to_numeric(tail["MD"], errors="coerce").to_numpy(np.float64)
    dm = np.diff(md)
    ds = np.diff(tvt) + np.diff(z)
    finite = np.isfinite(dm) & np.isfinite(ds) & (dm > 0.0)
    if int(finite.sum()) < 3:
        return 0.0
    return float(np.median(ds[finite] / dm[finite]))


@njit(cache=True)
def _interp1(grid: np.ndarray, value: float, vmin: float, step: float) -> float:
    i = int((value - vmin) / step)
    if i < 0:
        return grid[0]
    n = len(grid) - 1
    if i >= n:
        return grid[n]
    t = (value - vmin) / step - i
    return grid[i] * (1.0 - t) + grid[i + 1] * t


@njit(cache=True)
def _history_mean_std(values: np.ndarray, end: int, window: int, floor: float) -> tuple[float, float]:
    start = max(0, end - window)
    count = end - start
    if count < 3:
        return 0.0, floor
    mean = 0.0
    for index in range(start, end):
        mean += values[index]
    mean /= count
    variance = 0.0
    for index in range(start, end):
        diff = values[index] - mean
        variance += diff * diff
    variance /= count
    return mean, max(np.sqrt(variance), floor)


@njit(cache=True)
def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    total = 0.0
    for index in range(len(weights)):
        total += weights[index]
    if total <= 0.0:
        return values[order[len(order) // 2]]
    target = min(max(quantile, 0.0), 1.0) * total
    cumulative = 0.0
    for order_index in range(len(order)):
        index = order[order_index]
        cumulative += weights[index]
        if cumulative >= target:
            return values[index]
    return values[order[-1]]


@njit(cache=True)
def _adaptive_outlier_mixture_likpf_allseeds(
    md_v: np.ndarray,
    z_v: np.ndarray,
    gr_v: np.ndarray,
    gg: np.ndarray,
    vmin: float,
    step: float,
    gs: float,
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
    epsilon: float,
    outlier_uniform_likelihood: float,
    innovation_threshold: float,
    change_point_window: int,
    change_point_threshold: float,
    novelty_short_window: int,
    novelty_long_window: int,
    novelty_threshold: float,
    ess_ratio_threshold: float,
    max_weight_threshold: float,
    min_corroborating_signals: int,
    history_scale_floor: float,
    record_intervals: int,
    interval_stride: int,
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
]:
    n = len(md_v)
    preds = np.empty((n_seeds, n), dtype=np.float64)
    log_likelihoods = np.empty(n_seeds, dtype=np.float64)
    ess_accum = np.zeros(n, dtype=np.float64)
    resampled_accum = np.zeros(n, dtype=np.float64)
    gate_accum = np.zeros(n, dtype=np.float64)
    mixture_accum = np.zeros(n, dtype=np.float64)
    innovation_accum = np.zeros(n, dtype=np.float64)
    change_accum = np.zeros(n, dtype=np.float64)
    novelty_accum = np.zeros(n, dtype=np.float64)
    preupdate_ess_accum = np.zeros(n, dtype=np.float64)
    preupdate_max_weight_accum = np.zeros(n, dtype=np.float64)
    interval_p05 = np.full((n_seeds, n), np.nan, dtype=np.float64)
    interval_p95 = np.full((n_seeds, n), np.nan, dtype=np.float64)
    tmax = vmin + len(gg) * step
    for seed_index in range(n_seeds):
        np.random.seed((seed_base + seed_index) % (2**31 - 1))
        pos = np.empty(n_particles, dtype=np.float64)
        rate = np.empty(n_particles, dtype=np.float64)
        weights = np.empty(n_particles, dtype=np.float64)
        for particle_index in range(n_particles):
            pos[particle_index] = last_surface + init_spread * np.random.randn()
            rate[particle_index] = init_rate + 0.01 * np.random.randn()
            weights[particle_index] = 1.0 / n_particles
        log_lik = 0.0
        prev_md = md_v[0] - 1.0
        for row_index in range(n):
            delta_md = md_v[row_index] - prev_md
            if delta_md < 1.0:
                delta_md = 1.0
            for particle_index in range(n_particles):
                rate[particle_index] = momentum * rate[particle_index] + velocity_noise * np.random.randn()
                pos[particle_index] += rate[particle_index] * delta_md + position_noise * np.random.randn()
                tvt_particle = pos[particle_index] - z_v[row_index]
                if tvt_particle < vmin - 100.0:
                    tvt_particle = vmin - 100.0
                if tvt_particle > tmax + 100.0:
                    tvt_particle = tmax + 100.0
                pos[particle_index] = tvt_particle + z_v[row_index]

            expected_gr_mean = 0.0
            inverse_ess_before = 0.0
            max_weight_before = 0.0
            for particle_index in range(n_particles):
                expected_gr = _interp1(gg, pos[particle_index] - z_v[row_index], vmin, step)
                expected_gr_mean += weights[particle_index] * expected_gr
                inverse_ess_before += weights[particle_index] * weights[particle_index]
                if weights[particle_index] > max_weight_before:
                    max_weight_before = weights[particle_index]
            preupdate_ess_ratio = 1.0 / max(inverse_ess_before * n_particles, 1.0e-12)
            innovation = abs(gr_v[row_index] - expected_gr_mean) / max(gs, 1.0e-6)
            history_mean, history_std = _history_mean_std(
                gr_v,
                row_index,
                change_point_window,
                history_scale_floor,
            )
            change_point = abs(gr_v[row_index] - history_mean) / history_std
            short_mean, _ = _history_mean_std(
                gr_v,
                row_index,
                novelty_short_window,
                history_scale_floor,
            )
            long_mean, long_std = _history_mean_std(
                gr_v,
                row_index,
                novelty_long_window,
                history_scale_floor,
            )
            novelty = abs(short_mean - long_mean) / long_std
            corroborating = 0
            if change_point >= change_point_threshold:
                corroborating += 1
            if novelty >= novelty_threshold:
                corroborating += 1
            if preupdate_ess_ratio <= ess_ratio_threshold:
                corroborating += 1
            if max_weight_before >= max_weight_threshold:
                corroborating += 1
            gated = innovation >= innovation_threshold and corroborating >= min_corroborating_signals

            avg_likelihood = 0.0
            for particle_index in range(n_particles):
                expected_gr = _interp1(gg, pos[particle_index] - z_v[row_index], vmin, step)
                residual = (gr_v[row_index] - expected_gr) / gs
                residual2 = residual * residual
                if residual2 > 600.0:
                    residual2 = 600.0
                gaussian_likelihood = np.exp(-0.5 * residual2)
                if gaussian_likelihood < 1.0e-300:
                    gaussian_likelihood = 1.0e-300
                # The gate-off branch is intentionally the byte-for-byte exp072
                # Gaussian update. A uniform GR density is constant across all
                # particle states, so the gate-on outlier term is state-neutral.
                if gated:
                    likelihood = (
                        (1.0 - epsilon) * gaussian_likelihood
                        + epsilon * outlier_uniform_likelihood
                    )
                else:
                    likelihood = gaussian_likelihood
                avg_likelihood += weights[particle_index] * likelihood
                weights[particle_index] *= likelihood
            if avg_likelihood < 1.0e-300:
                avg_likelihood = 1.0e-300
            log_lik += np.log(avg_likelihood)

            weight_sum = 0.0
            for particle_index in range(n_particles):
                weight_sum += weights[particle_index]
            if weight_sum > 0.0:
                for particle_index in range(n_particles):
                    weights[particle_index] /= weight_sum
            else:
                for particle_index in range(n_particles):
                    weights[particle_index] = 1.0 / n_particles

            if record_intervals == 1 and (row_index % interval_stride == 0 or gated):
                tvt_values = np.empty(n_particles, dtype=np.float64)
                for particle_index in range(n_particles):
                    tvt_values[particle_index] = pos[particle_index] - z_v[row_index]
                interval_p05[seed_index, row_index] = _weighted_quantile(tvt_values, weights, 0.05)
                interval_p95[seed_index, row_index] = _weighted_quantile(tvt_values, weights, 0.95)

            inverse_ess_after = 0.0
            for particle_index in range(n_particles):
                inverse_ess_after += weights[particle_index] * weights[particle_index]
            ess = 1.0 / max(inverse_ess_after, 1.0e-12)
            ess_accum[row_index] += ess
            gate_accum[row_index] += 1.0 if gated else 0.0
            mixture_accum[row_index] += 1.0 if gated else 0.0
            innovation_accum[row_index] += innovation
            change_accum[row_index] += change_point
            novelty_accum[row_index] += novelty
            preupdate_ess_accum[row_index] += preupdate_ess_ratio
            preupdate_max_weight_accum[row_index] += max_weight_before

            if ess < resample_threshold * n_particles:
                cumulative = np.empty(n_particles, dtype=np.float64)
                cumulative_weight = 0.0
                for particle_index in range(n_particles):
                    cumulative_weight += weights[particle_index]
                    cumulative[particle_index] = cumulative_weight
                u0 = np.random.uniform(0.0, 1.0 / n_particles)
                new_pos = np.empty(n_particles, dtype=np.float64)
                new_rate = np.empty(n_particles, dtype=np.float64)
                cumulative_index = 0
                for particle_index in range(n_particles):
                    draw = u0 + particle_index / n_particles
                    while cumulative_index < n_particles - 1 and cumulative[cumulative_index] < draw:
                        cumulative_index += 1
                    new_pos[particle_index] = pos[cumulative_index] + resample_pos_noise * np.random.randn()
                    new_rate[particle_index] = rate[cumulative_index] + resample_velocity_noise * np.random.randn()
                for particle_index in range(n_particles):
                    pos[particle_index] = new_pos[particle_index]
                    rate[particle_index] = new_rate[particle_index]
                    weights[particle_index] = 1.0 / n_particles
                resampled_accum[row_index] += 1.0

            estimate = 0.0
            for particle_index in range(n_particles):
                estimate += weights[particle_index] * (pos[particle_index] - z_v[row_index])
            preds[seed_index, row_index] = estimate
            prev_md = md_v[row_index]
        log_likelihoods[seed_index] = log_lik
    return (
        preds,
        log_likelihoods,
        ess_accum / n_seeds,
        resampled_accum / n_seeds,
        gate_accum / n_seeds,
        mixture_accum / n_seeds,
        innovation_accum / n_seeds,
        change_accum / n_seeds,
        novelty_accum / n_seeds,
        preupdate_ess_accum / n_seeds,
        preupdate_max_weight_accum / n_seeds,
        interval_p05,
        interval_p95,
    )


def seed_weights_for_temperature(log_likelihoods: np.ndarray, temperature: float) -> np.ndarray:
    centered = log_likelihoods.astype(np.float64) - float(np.max(log_likelihoods))
    weights = np.exp(centered / max(float(temperature), 1e-6))
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0 or not np.isfinite(weight_sum):
        return np.full(len(log_likelihoods), 1.0 / max(len(log_likelihoods), 1), dtype=np.float32)
    return (weights / weight_sum).astype(np.float32)


def format_scale(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def systematic_resample(
    rng: np.random.Generator,
    pos: np.ndarray,
    vel: np.ndarray,
    weights: np.ndarray,
    pos_noise: float,
    vel_noise: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(pos)
    cdf = np.cumsum(weights)
    cdf[-1] = 1.0
    positions = rng.uniform(0.0, 1.0 / n) + np.arange(n, dtype=np.float64) / n
    idx = np.searchsorted(cdf, positions, side="left")
    return (
        pos[idx] + pos_noise * rng.standard_normal(n),
        vel[idx] + vel_noise * rng.standard_normal(n),
    )


def robust_affine_fit(
    x: np.ndarray,
    y: np.ndarray,
    *,
    trim_quantile: float,
    iterations: int,
) -> tuple[float, float, np.ndarray]:
    finite = np.isfinite(x) & np.isfinite(y)
    mask = finite.copy()
    if int(mask.sum()) < 2:
        return 1.0, 0.0, mask
    for _ in range(max(1, int(iterations))):
        xm = x[mask].astype(np.float64)
        ym = y[mask].astype(np.float64)
        x_center = float(np.mean(xm))
        y_center = float(np.mean(ym))
        denom = float(np.sum((xm - x_center) ** 2))
        if denom <= 1.0e-12:
            return 1.0, 0.0, mask
        slope = float(np.sum((xm - x_center) * (ym - y_center)) / denom)
        intercept = y_center - slope * x_center
        residual = np.abs(y - (slope * x + intercept))
        finite_residual = residual[finite & np.isfinite(residual)]
        if len(finite_residual) < 2:
            break
        cutoff = float(np.nanquantile(finite_residual, float(trim_quantile)))
        next_mask = finite & (residual <= cutoff)
        if int(next_mask.sum()) < 2 or np.array_equal(next_mask, mask):
            break
        mask = next_mask
    xm = x[mask].astype(np.float64)
    ym = y[mask].astype(np.float64)
    x_center = float(np.mean(xm))
    y_center = float(np.mean(ym))
    denom = float(np.sum((xm - x_center) ** 2))
    if denom <= 1.0e-12:
        return 1.0, 0.0, mask
    slope = float(np.sum((xm - x_center) * (ym - y_center)) / denom)
    intercept = y_center - slope * x_center
    return slope, intercept, mask


def affine_calibration_for_holdout(
    holdout: PrefixHoldout,
    tw_tvt: np.ndarray,
    raw_tw_gr: np.ndarray,
    raw_hw_gr: np.ndarray,
    spec: GrFilterSpec,
) -> tuple[np.ndarray, dict[str, Any]]:
    params = spec.params
    min_points = int(params.get("min_prefix_points", 40))
    min_tw_std = float(params.get("min_typewell_gr_std", 5.0))
    slope_min = float(params.get("slope_min", 0.25))
    slope_max = float(params.get("slope_max", 4.0))
    max_prefix_rmse = float(params.get("max_prefix_rmse", 60.0))
    trim_quantile = float(params.get("trim_quantile", 0.90))
    iterations = int(params.get("robust_iterations", 2))

    eval_start = int(holdout.eval_index[0])
    prefix = holdout.masked.iloc[:eval_start]
    finite = prefix["TVT_input"].notna() & prefix["GR"].notna()
    prefix_pos = prefix.index.to_numpy(np.int64)
    tvt = pd.to_numeric(prefix.loc[finite, "TVT_input"], errors="coerce").to_numpy(np.float64)
    gr = raw_hw_gr[prefix_pos[finite.to_numpy()]].astype(np.float64)
    tw_at_prefix = np.interp(tvt, tw_tvt, raw_tw_gr).astype(np.float64)
    fit_finite = np.isfinite(tvt) & np.isfinite(gr) & np.isfinite(tw_at_prefix)

    metadata: dict[str, Any] = {
        "effective_kind": "affine_calibrated",
        "fit_scope": "known_prefix_only",
        "fallback": False,
        "fallback_reason": None,
        "prefix_points": int(fit_finite.sum()),
        "min_prefix_points": min_points,
        "min_typewell_gr_std": min_tw_std,
        "slope_min": slope_min,
        "slope_max": slope_max,
        "max_prefix_rmse": max_prefix_rmse,
        "trim_quantile": trim_quantile,
        "robust_iterations": iterations,
    }
    if int(fit_finite.sum()) < min_points:
        metadata.update({"fallback": True, "fallback_reason": "prefix_too_short"})
        return raw_hw_gr.astype(np.float64), metadata

    x = tw_at_prefix[fit_finite]
    y = gr[fit_finite]
    typewell_std = float(np.nanstd(x))
    metadata["typewell_gr_std"] = typewell_std
    if not np.isfinite(typewell_std) or typewell_std < min_tw_std:
        metadata.update({"fallback": True, "fallback_reason": "low_typewell_gr_std"})
        return raw_hw_gr.astype(np.float64), metadata

    slope, intercept, mask = robust_affine_fit(
        x,
        y,
        trim_quantile=trim_quantile,
        iterations=iterations,
    )
    pred = slope * x + intercept
    residual = y - pred
    prefix_rmse = float(np.sqrt(np.mean(residual[mask] * residual[mask]))) if mask.any() else np.inf
    metadata.update(
        {
            "slope": slope,
            "intercept": intercept,
            "used_points": int(mask.sum()),
            "typewell_gr_std": typewell_std,
            "prefix_rmse": prefix_rmse,
            "prefix_mae": float(np.mean(np.abs(residual[mask]))) if mask.any() else None,
        }
    )
    if not np.isfinite(slope) or not np.isfinite(intercept) or slope <= 0.0:
        metadata.update({"fallback": True, "fallback_reason": "invalid_slope"})
        return raw_hw_gr.astype(np.float64), metadata
    if slope < slope_min or slope > slope_max:
        metadata.update({"fallback": True, "fallback_reason": "extreme_slope"})
        return raw_hw_gr.astype(np.float64), metadata
    if not np.isfinite(prefix_rmse) or prefix_rmse > max_prefix_rmse:
        metadata.update({"fallback": True, "fallback_reason": "high_prefix_rmse"})
        return raw_hw_gr.astype(np.float64), metadata

    calibrated = (raw_hw_gr.astype(np.float64) - intercept) / max(slope, 1.0e-6)
    return calibrated.astype(np.float64), metadata


def structural_prior_for_holdout(
    holdout: PrefixHoldout,
    spec: GrFilterSpec,
    eval_md: np.ndarray,
) -> StructuralPrior:
    if spec.transition != "prefix_structural":
        return StructuralPrior(
            active=False,
            expected_tvt=np.full(len(eval_md), np.nan, dtype=np.float32),
            sigma=np.nan,
            weight=0.0,
            metadata={"transition": spec.transition, "active": False},
        )
    params = get_nested({"params": spec.transition_params}, "params") or {}
    min_points = int(params.get("min_prefix_points", 80))
    tail_points = int(params.get("tail_points", 256))
    sigma_min = float(params.get("sigma_min", 40.0))
    sigma_max = float(params.get("sigma_max", 180.0))
    sigma_default = float(params.get("sigma_default", 90.0))
    weight = float(params.get("weight", 0.20))

    eval_start = int(holdout.eval_index[0])
    prefix = holdout.masked.iloc[:eval_start].tail(tail_points).copy()
    required = prefix["TVT_input"].notna() & prefix["MD"].notna() & prefix["Z"].notna()
    if int(required.sum()) < min_points:
        return StructuralPrior(
            active=False,
            expected_tvt=np.full(len(eval_md), np.nan, dtype=np.float32),
            sigma=sigma_default,
            weight=0.0,
            metadata={
                "transition": spec.transition,
                "active": False,
                "fallback_reason": "prefix_too_short",
                "prefix_points": int(required.sum()),
            },
        )
    md = pd.to_numeric(prefix.loc[required, "MD"], errors="coerce").to_numpy(np.float64)
    tvt = pd.to_numeric(prefix.loc[required, "TVT_input"], errors="coerce").to_numpy(np.float64)
    z = pd.to_numeric(prefix.loc[required, "Z"], errors="coerce").to_numpy(np.float64)
    surface = tvt + z
    finite = np.isfinite(md) & np.isfinite(surface)
    if int(finite.sum()) < min_points:
        return StructuralPrior(
            active=False,
            expected_tvt=np.full(len(eval_md), np.nan, dtype=np.float32),
            sigma=sigma_default,
            weight=0.0,
            metadata={
                "transition": spec.transition,
                "active": False,
                "fallback_reason": "nonfinite_prefix_surface",
                "prefix_points": int(finite.sum()),
            },
        )
    slope, intercept, mask = robust_affine_fit(
        md[finite],
        surface[finite],
        trim_quantile=float(params.get("trim_quantile", 0.90)),
        iterations=int(params.get("robust_iterations", 2)),
    )
    prefix_pred = slope * md[finite] + intercept
    residual = surface[finite] - prefix_pred
    resid_sigma = float(np.nanstd(residual[mask])) if mask.any() else sigma_default
    sigma = float(np.clip(max(resid_sigma, sigma_default), sigma_min, sigma_max))
    eval_rows = holdout.masked.loc[holdout.eval_index]
    eval_z = numeric_array(eval_rows, "Z").astype(np.float64)
    expected_surface = slope * eval_md.astype(np.float64) + intercept
    expected_tvt = (expected_surface - eval_z).astype(np.float32)
    return StructuralPrior(
        active=True,
        expected_tvt=expected_tvt,
        sigma=sigma,
        weight=weight,
        metadata={
            "transition": spec.transition,
            "active": True,
            "fit_scope": "known_prefix_tail_only",
            "prefix_points": int(finite.sum()),
            "used_points": int(mask.sum()),
            "tail_points": tail_points,
            "surface_slope": slope,
            "surface_intercept": intercept,
            "surface_residual_sigma": resid_sigma,
            "sigma": sigma,
            "weight": weight,
        },
    )


def filtered_observations(
    holdout: PrefixHoldout,
    spec: GrFilterSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    hw = holdout.masked
    tw = holdout.typewell.sort_values("TVT").reset_index(drop=True)
    tw_tvt = numeric_array(tw, "TVT").astype(np.float64)
    raw_tw_gr = fill_numeric(tw["GR"], float(np.nanmean(numeric_array(tw, "GR")))).astype(
        np.float64
    )
    raw_hw_gr = fill_numeric(hw["GR"], float(np.nanmean(raw_tw_gr))).astype(np.float64)
    if spec.kind == "raw":
        filtered_tw_gr = raw_tw_gr.astype(np.float64)
        filtered_hw_gr = raw_hw_gr.astype(np.float64)
        tw_meta = {"effective_kind": "raw"}
        hw_meta = {"effective_kind": "raw"}
    elif spec.kind == "affine_calibrated":
        filtered_tw_gr = raw_tw_gr.astype(np.float64)
        filtered_hw_gr, hw_meta = affine_calibration_for_holdout(
            holdout,
            tw_tvt,
            raw_tw_gr,
            raw_hw_gr,
            spec,
        )
        tw_meta = {"effective_kind": "raw_typewell_reference"}
    else:
        filtered_tw_gr, tw_meta = apply_gr_filter(raw_tw_gr, spec)
        filtered_hw_gr, hw_meta = apply_gr_filter(raw_hw_gr, spec)
    metadata = {
        "variant": spec.name,
        "kind": spec.kind,
        "transition": spec.transition,
        "params": spec.params,
        "transition_params": spec.transition_params,
        "typewell_filter": tw_meta,
        "horizontal_filter": hw_meta,
    }
    return (
        tw_tvt,
        filtered_tw_gr.astype(np.float64),
        filtered_hw_gr.astype(np.float64),
        raw_hw_gr,
        metadata,
    )


def run_pf_for_holdout(
    holdout: PrefixHoldout,
    *,
    mixture: OutlierMixtureSpec,
    record_intervals: bool,
    config: dict[str, Any],
) -> PfRun:
    runtime = get_nested(config, "model.runtime") or {}
    gate = get_nested(config, "model.gate") or {}
    interval_audit = get_nested(config, "model.interval_audit") or {}
    n_particles = int(runtime.get("particles", 500))
    seed_count = int(runtime.get("seed_count", 128))
    resample_threshold = float(runtime.get("resample_threshold", 0.5))
    init_spread = float(runtime.get("init_spread", 4.5))
    velocity_noise = float(runtime.get("velocity_noise", 0.002))
    position_noise = float(runtime.get("position_noise", 0.005))
    resample_pos_noise = float(runtime.get("resample_pos_noise", 0.10))
    resample_velocity_noise = float(runtime.get("resample_velocity_noise", 0.001))

    raw_spec = GrFilterSpec(
        name="raw",
        kind="raw",
        params={},
        transition="classic",
        transition_params={},
    )
    hw = holdout.masked
    tw_tvt, tw_gr, hw_gr, _, metadata = filtered_observations(holdout, raw_spec)
    eval_rows = hw.loc[holdout.eval_index].copy()
    prefix = hw.iloc[: int(holdout.eval_index[0])]
    sigma = gr_sigma(
        horizontal=hw,
        filtered_horizontal_gr=hw_gr,
        eval_start=int(holdout.eval_index[0]),
        tw_tvt=tw_tvt,
        filtered_typewell_gr=tw_gr,
        config=config,
    )
    init_vel = initial_surface_velocity(prefix)
    last_prefix = prefix.iloc[-1]
    last_surface = float(last_prefix["TVT_input"]) + float(last_prefix["Z"])
    grid_step = float(runtime.get("grid_step", 0.2))
    tmin = float(np.nanmin(tw_tvt))
    tmax = float(np.nanmax(tw_tvt))
    tvt_grid = np.arange(tmin, tmax + grid_step, grid_step, dtype=np.float64)
    gr_grid = np.interp(tvt_grid, tw_tvt, tw_gr).astype(np.float64)
    observed_gr = hw_gr[holdout.eval_index].astype(np.float64)
    if np.any(observed_gr < mixture.uniform_gr_min) or np.any(observed_gr > mixture.uniform_gr_max):
        observed_min = float(np.min(observed_gr))
        observed_max = float(np.max(observed_gr))
        raise ValueError(
            "Evaluation GR is outside the fixed state-neutral Uniform support: "
            f"observed=[{observed_min}, {observed_max}], "
            f"support=[{mixture.uniform_gr_min}, {mixture.uniform_gr_max}]"
        )
    uniform_width = mixture.uniform_gr_max - mixture.uniform_gr_min
    outlier_uniform_likelihood = float(np.sqrt(2.0 * np.pi) * sigma / uniform_width)
    result = _adaptive_outlier_mixture_likpf_allseeds(
        numeric_array(eval_rows, "MD").astype(np.float64),
        numeric_array(eval_rows, "Z").astype(np.float64),
        observed_gr,
        gr_grid,
        tmin,
        grid_step,
        max(float(sigma), 1.0e-6),
        last_surface,
        init_vel,
        n_particles,
        seed_count,
        stable_seed(EXPERIMENT_NAME, holdout.well, mixture.name, "public_likpf"),
        float(runtime.get("momentum", 0.998)),
        velocity_noise,
        position_noise,
        resample_pos_noise,
        resample_velocity_noise,
        resample_threshold,
        init_spread,
        float(mixture.epsilon),
        outlier_uniform_likelihood,
        float(gate.get("innovation_threshold", 2.5)),
        int(gate.get("change_point_window", 32)),
        float(gate.get("change_point_threshold", 3.0)),
        int(gate.get("novelty_short_window", 8)),
        int(gate.get("novelty_long_window", 64)),
        float(gate.get("novelty_threshold", 2.5)),
        float(gate.get("ess_ratio_threshold", 0.25)),
        float(gate.get("max_weight_threshold", 0.10)),
        int(gate.get("min_corroborating_signals", 1)),
        float(gate.get("history_scale_floor", 5.0)),
        1 if record_intervals and bool(interval_audit.get("enabled", True)) else 0,
        int(interval_audit.get("row_stride", 64)),
    )
    (
        preds,
        log_likelihoods,
        ess_mean,
        resampled_rate,
        gate_rate,
        mixture_rate,
        innovation,
        change_point,
        novelty,
        preupdate_ess_ratio,
        preupdate_max_weight,
        interval_p05,
        interval_p95,
    ) = result
    metadata = dict(metadata)
    if not np.array_equal(gate_rate, mixture_rate):
        raise RuntimeError("Mixture application rate must exactly equal the target-free gate rate.")
    metadata["pf_algorithm"] = "adaptive_outlier_mixture_public_surface_likelihood_pf_numba"
    metadata["public_like"] = {
        "last_surface": last_surface,
        "init_surface_rate": init_vel,
        "grid_step": grid_step,
        "particles": n_particles,
        "seed_count": seed_count,
        "sigma": sigma,
        "mixture": {
            "component": mixture.component,
            "epsilon": float(mixture.epsilon),
            "uniform_gr_min": float(mixture.uniform_gr_min),
            "uniform_gr_max": float(mixture.uniform_gr_max),
            "uniform_likelihood": outlier_uniform_likelihood,
        },
        "record_intervals": bool(record_intervals),
        "gate": to_jsonable(gate),
    }
    return PfRun(
        preds=preds.astype(np.float32),
        log_likelihoods=log_likelihoods,
        ess_mean_by_row=ess_mean.astype(np.float32),
        resampled_by_row=resampled_rate.astype(np.float32),
        gate_rate_by_row=gate_rate.astype(np.float32),
        mixture_rate_by_row=mixture_rate.astype(np.float32),
        innovation_by_row=innovation.astype(np.float32),
        change_point_by_row=change_point.astype(np.float32),
        novelty_by_row=novelty.astype(np.float32),
        preupdate_ess_ratio_by_row=preupdate_ess_ratio.astype(np.float32),
        preupdate_max_weight_by_row=preupdate_max_weight.astype(np.float32),
        interval_p05_by_seed=interval_p05.astype(np.float32),
        interval_p95_by_seed=interval_p95.astype(np.float32),
        sigma=float(sigma),
        filter_metadata=metadata,
    )


def nearest_index(values: np.ndarray, target: float) -> int:
    idx = int(np.searchsorted(values, target, side="left"))
    if idx >= len(values):
        return len(values) - 1
    if idx > 0 and abs(float(values[idx - 1]) - target) <= abs(float(values[idx]) - target):
        return idx - 1
    return idx


def beam_search_for_holdout(
    holdout: PrefixHoldout,
    spec: GrFilterSpec,
    config: dict[str, Any],
) -> np.ndarray:
    beam_cfg = get_nested(config, "model.beam") or {}
    beam_size = int(beam_cfg.get("beam_size", 14))
    move_radius = int(beam_cfg.get("move_radius", 2))
    move_cost = float(beam_cfg.get("move_cost", 16.0))
    error_scale = float(beam_cfg.get("error_scale", 120.0))
    post_smooth_window = int(beam_cfg.get("post_smooth_window", 1))

    tw_tvt, tw_gr, hw_gr, _, _ = filtered_observations(holdout, spec)
    if post_smooth_window > 1:
        tw_gr = rolling_mean(tw_gr, post_smooth_window).astype(np.float64)
        hw_gr = rolling_mean(hw_gr, post_smooth_window).astype(np.float64)
    gr = hw_gr[holdout.eval_index].astype(np.float64)
    eval_rows = holdout.masked.loc[holdout.eval_index]
    md = numeric_array(eval_rows, "MD").astype(np.float64)
    structural_prior = structural_prior_for_holdout(holdout, spec, md)

    start_idx = nearest_index(tw_tvt, holdout.last_known_tvt)
    active: dict[int, tuple[float, list[int]]] = {start_idx: (0.0, [])}
    for row_pos, row_gr in enumerate(gr):
        candidates: dict[int, tuple[float, list[int]]] = {}
        for idx, (cost, path) in active.items():
            for delta in range(-move_radius, move_radius + 1):
                next_idx = int(np.clip(idx + delta, 0, len(tw_tvt) - 1))
                gr_cost = ((float(row_gr) - float(tw_gr[next_idx])) ** 2) / max(
                    error_scale,
                    1e-6,
                )
                prior_cost = 0.0
                if structural_prior.active and np.isfinite(structural_prior.expected_tvt[row_pos]):
                    prior_residual = (
                        float(tw_tvt[next_idx]) - float(structural_prior.expected_tvt[row_pos])
                    ) / max(structural_prior.sigma, 1.0e-6)
                    prior_cost = 0.5 * structural_prior.weight * min(
                        prior_residual * prior_residual,
                        600.0,
                    )
                total = cost + gr_cost + move_cost * abs(delta) + prior_cost
                previous = candidates.get(next_idx)
                if previous is None or total < previous[0]:
                    candidates[next_idx] = (total, [*path, next_idx])
        kept = sorted(candidates.items(), key=lambda item: item[1][0])[:beam_size]
        active = {idx: value for idx, value in kept}
    if not active:
        return np.full(len(holdout.eval_index), holdout.last_known_tvt, dtype=np.float32)
    _, (_, best_path) = min(active.items(), key=lambda item: item[1][0])
    return tw_tvt[np.asarray(best_path, dtype=np.int64)].astype(np.float32)


def score_prediction(pred: np.ndarray, true: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(pred) & np.isfinite(true)
    if not finite.any():
        return {
            "rows": 0,
            "coverage": 0.0,
            "rmse": None,
            "mae": None,
            "within10": None,
            "bias": None,
        }
    err = pred[finite].astype(np.float64) - true[finite].astype(np.float64)
    return {
        "rows": int(finite.sum()),
        "coverage": float(finite.mean()),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mae": float(np.mean(np.abs(err))),
        "within10": float(np.mean(np.abs(err) <= 10.0)),
        "bias": float(np.mean(err)),
    }


def distance_bucket(md_since: pd.Series | np.ndarray) -> pd.Categorical:
    return pd.cut(
        pd.to_numeric(pd.Series(md_since), errors="coerce"),
        bins=[-np.inf, 50.0, 100.0, 250.0, 500.0, 1000.0, np.inf],
        labels=["000_050", "050_100", "100_250", "250_500", "500_1000", "1000_plus"],
        include_lowest=True,
    )


def candidate_columns(frame: pd.DataFrame) -> list[str]:
    excluded_suffixes = ("_diag", "_name")
    prefixes = ("pf_", "beam_", "exp072_", "oracle_")
    return [
        column
        for column in frame.columns
        if column.startswith(prefixes) and not column.endswith(excluded_suffixes)
    ]


def is_oracle_candidate(column: str) -> bool:
    return "_oracle" in column or column.startswith("oracle_")


def path_jump_rate_for_candidate(
    frame: pd.DataFrame,
    column: str,
    threshold_ft: float,
) -> float | None:
    jumps = 0
    total = 0
    for _, group in frame.sort_values(["well", "row_idx"]).groupby("well", sort=False):
        values = pd.to_numeric(group[column], errors="coerce").to_numpy(np.float64)
        finite = np.isfinite(values)
        if finite.sum() < 2:
            continue
        diffs = np.abs(np.diff(values[finite]))
        jumps += int(np.sum(diffs > threshold_ft))
        total += int(len(diffs))
    if total == 0:
        return None
    return float(jumps / total)


def compute_candidate_metrics(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    true = numeric_array(frame, "true_tvt")
    primary = str(get_nested(config, "audit.primary_baseline") or "pf_raw_lik_mean")
    baseline = score_prediction(numeric_array(frame, primary), true) if primary in frame else None
    threshold = float(get_nested(config, "audit.path_jump_threshold_ft") or 8.0)
    rows: list[dict[str, Any]] = []
    for column in candidate_columns(frame):
        pred = numeric_array(frame, column)
        score = score_prediction(pred, true)
        delta = None
        if baseline and score["rmse"] is not None and baseline["rmse"] is not None:
            delta = float(score["rmse"] - baseline["rmse"])
        rows.append(
            {
                "candidate": column,
                "is_oracle_diagnostic": is_oracle_candidate(column),
                **score,
                "delta_rmse_vs_primary_baseline": delta,
                "path_jump_rate": path_jump_rate_for_candidate(frame, column, threshold),
            }
        )
    return pd.DataFrame(rows).sort_values(["rmse", "candidate"], na_position="last")


def compute_filter_delta_metrics(frame: pd.DataFrame, filters: list[GrFilterSpec]) -> pd.DataFrame:
    true = numeric_array(frame, "true_tvt")
    output_columns = [
        "family",
        "filter",
        "filter_kind",
        "candidate",
        "raw_candidate",
        "rows",
        "rmse",
        "mae",
        "within10",
        "bias",
        "raw_rmse",
        "delta_rmse_vs_raw_family",
        "row_abs_diff_mean_vs_raw",
        "row_diff_rmse_vs_raw",
        "changed_rows_vs_raw",
    ]
    specs = [
        ("pf_lik_mean", "pf_raw_lik_mean", "pf_{name}_lik_mean"),
        ("pf_best_seed", "pf_raw_best_seed", "pf_{name}_best_seed"),
        ("pf_top3_oracle", "pf_raw_top3_oracle", "pf_{name}_top3_oracle"),
        ("beam_top1", "beam_raw_top1", "beam_{name}_top1"),
    ]
    rows: list[dict[str, Any]] = []
    for family, raw_column, template in specs:
        if raw_column not in frame:
            continue
        raw_score = score_prediction(numeric_array(frame, raw_column), true)
        raw_values = numeric_array(frame, raw_column)
        for spec in filters:
            if spec.name == "raw":
                continue
            column = template.format(name=spec.name)
            if column not in frame:
                continue
            score = score_prediction(numeric_array(frame, column), true)
            values = numeric_array(frame, column)
            diff = values.astype(np.float64) - raw_values.astype(np.float64)
            finite = np.isfinite(diff)
            delta = None
            if score["rmse"] is not None and raw_score["rmse"] is not None:
                delta = float(score["rmse"] - raw_score["rmse"])
            rows.append(
                {
                    "family": family,
                    "filter": spec.name,
                    "filter_kind": spec.kind,
                    "candidate": column,
                    "raw_candidate": raw_column,
                    **score,
                    "raw_rmse": raw_score["rmse"],
                    "delta_rmse_vs_raw_family": delta,
                    "row_abs_diff_mean_vs_raw": (
                        float(np.mean(np.abs(diff[finite]))) if finite.any() else None
                    ),
                    "row_diff_rmse_vs_raw": (
                        float(np.sqrt(np.mean(diff[finite] * diff[finite])))
                        if finite.any()
                        else None
                    ),
                    "changed_rows_vs_raw": int(np.sum(np.abs(diff[finite]) > 1.0e-6))
                    if finite.any()
                    else 0,
                }
            )
    if not rows:
        primary = "pf_raw_scale_5" if "pf_raw_scale_5" in frame else "pf_raw_lik_mean"
        raw_score = score_prediction(numeric_array(frame, primary), true) if primary in frame else {}
        raw_values = numeric_array(frame, primary) if primary in frame else np.full(len(frame), np.nan)
        for column in candidate_columns(frame):
            if column == primary or is_oracle_candidate(column):
                continue
            score = score_prediction(numeric_array(frame, column), true)
            values = numeric_array(frame, column)
            diff = values.astype(np.float64) - raw_values.astype(np.float64)
            finite = np.isfinite(diff)
            rows.append(
                {
                    "family": "raw_public_control",
                    "filter": "raw",
                    "filter_kind": "raw",
                    "candidate": column,
                    "raw_candidate": primary,
                    **score,
                    "raw_rmse": raw_score.get("rmse"),
                    "delta_rmse_vs_raw_family": (
                        float(score["rmse"] - raw_score["rmse"])
                        if score["rmse"] is not None and raw_score.get("rmse") is not None
                        else None
                    ),
                    "row_abs_diff_mean_vs_raw": (
                        float(np.mean(np.abs(diff[finite]))) if finite.any() else None
                    ),
                    "row_diff_rmse_vs_raw": (
                        float(np.sqrt(np.mean(diff[finite] * diff[finite])))
                        if finite.any()
                        else None
                    ),
                    "changed_rows_vs_raw": int(np.sum(np.abs(diff[finite]) > 1.0e-6))
                    if finite.any()
                    else 0,
                }
            )
    if not rows:
        return pd.DataFrame(columns=output_columns)
    return pd.DataFrame(rows).sort_values(
        ["delta_rmse_vs_raw_family", "candidate"],
        na_position="last",
    )


def compute_bucket_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    true = numeric_array(frame, "true_tvt")
    buckets = distance_bucket(frame["md_since"])
    rows: list[dict[str, Any]] = []
    for column in candidate_columns(frame):
        pred = numeric_array(frame, column)
        for bucket in pd.Series(buckets).cat.categories:
            mask = np.asarray(buckets == bucket, dtype=bool)
            if not mask.any():
                continue
            rows.append(
                {
                    "candidate": column,
                    "distance_bucket": str(bucket),
                    **score_prediction(pred[mask], true[mask]),
                }
            )
    return pd.DataFrame(rows)


def compute_by_well(frame: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    primary = str(get_nested(config, "audit.primary_baseline") or "pf_raw_lik_mean")
    rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well", sort=False):
        true = numeric_array(group, "true_tvt")
        base_rmse = None
        if primary in group:
            base_rmse = score_prediction(numeric_array(group, primary), true)["rmse"]
        for column in candidate_columns(group):
            score = score_prediction(numeric_array(group, column), true)
            delta = None
            if base_rmse is not None and score["rmse"] is not None:
                delta = float(score["rmse"] - base_rmse)
            rows.append(
                {
                    "well": str(well),
                    "candidate": column,
                    "eval_rows": int(len(group)),
                    "max_md_since": float(pd.to_numeric(group["md_since"], errors="coerce").max()),
                    **score,
                    "delta_rmse_vs_primary_baseline": delta,
                }
            )
    return pd.DataFrame(rows)


def compute_group_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    md_since = numeric_array(frame, "md_since")
    groups: dict[str, np.ndarray] = {
        "all": np.ones(len(frame), dtype=bool),
        "near_000_050": md_since <= 50.0,
        "mid_250_1000": (md_since >= 250.0) & (md_since < 1000.0),
        "longtail_1000_plus": md_since >= 1000.0,
        "very_longtail_2000_plus": md_since >= 2000.0,
    }
    rows: list[dict[str, Any]] = []
    true = numeric_array(frame, "true_tvt")
    for group_name, mask in groups.items():
        if not mask.any():
            continue
        for column in candidate_columns(frame):
            rows.append(
                {
                    "group": group_name,
                    "candidate": column,
                    **score_prediction(numeric_array(frame, column)[mask], true[mask]),
                }
            )
    return pd.DataFrame(rows)


def add_worst_well_regression(candidate_metrics: pd.DataFrame, by_well: pd.DataFrame) -> None:
    if by_well.empty:
        candidate_metrics["max_well_regression_vs_primary"] = np.nan
        return
    max_regression = by_well.groupby("candidate", observed=True)[
        "delta_rmse_vs_primary_baseline"
    ].max()
    candidate_metrics["max_well_regression_vs_primary"] = candidate_metrics["candidate"].map(
        max_regression
    )


def rowwise_oracle(
    frame: pd.DataFrame,
    columns: list[str],
    output_name: str,
) -> None:
    columns = [column for column in columns if column in frame.columns]
    if not columns:
        return
    true = numeric_array(frame, "true_tvt").astype(np.float64)
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
    err = np.abs(values - true[:, None])
    all_nan = ~np.isfinite(err).any(axis=1)
    err[~np.isfinite(err)] = np.inf
    best_pos = np.argmin(err, axis=1)
    pred = values[np.arange(len(frame)), best_pos]
    pred[all_nan] = np.nan
    frame[output_name] = pred.astype(np.float32)
    frame[f"{output_name}_name"] = np.asarray(columns, dtype=object)[best_pos]


def build_row_frame_for_holdout(
    holdout: PrefixHoldout,
    filters: list[GrFilterSpec],
    pf_outputs: dict[str, PfRun],
    beam_outputs: dict[str, np.ndarray],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    eval_rows = holdout.masked.loc[holdout.eval_index]
    md_since = (
        holdout.cache_md_since
        if holdout.cache_md_since is not None
        else numeric_array(eval_rows, "MD") - np.float32(holdout.last_known_md)
    )
    row_frame = pd.DataFrame(
        {
            "well": holdout.well,
            "row_idx": holdout.eval_index.astype(np.int32),
            "id": holdout.eval_ids.astype(str),
            "target": holdout.target_delta.astype(np.float32),
            "true_tvt": holdout.true_tvt.astype(np.float32),
            "last_known_tvt": np.float32(holdout.last_known_tvt),
            "last_known_md": np.float32(holdout.last_known_md),
            "md_since": md_since.astype(np.float32),
        }
    )
    for column in holdout.reference_candidates.columns:
        row_frame[f"exp072_{column}"] = pd.to_numeric(
            holdout.reference_candidates[column],
            errors="coerce",
        ).to_numpy(np.float32)

    top_k = int(get_nested(config, "model.runtime.topk_oracle") or 3)
    runtime = get_nested(config, "model.runtime") or {}
    scales = runtime.get("likelihood_scales") or [runtime.get("primary_scale", 5.0)]
    scales = [float(scale) for scale in scales]
    primary_scale = float(runtime.get("primary_scale", scales[0] if scales else 5.0))
    diagnostics: list[dict[str, Any]] = []
    for spec in filters:
        run = pf_outputs[spec.name]
        primary_weights = seed_weights_for_temperature(run.log_likelihoods, primary_scale)
        primary_weighted = (primary_weights[:, None] * run.preds).sum(axis=0)
        best_idx = int(np.argmax(run.log_likelihoods))
        top_idx = np.argsort(run.log_likelihoods)[::-1][:top_k]
        oracle = np.empty(len(row_frame), dtype=np.float32)
        truth = row_frame["true_tvt"].to_numpy(np.float32)
        for i in range(len(row_frame)):
            seed_values = run.preds[top_idx, i]
            oracle[i] = seed_values[np.argmin(np.abs(seed_values - truth[i]))]
        for scale in scales:
            scale_weights = seed_weights_for_temperature(run.log_likelihoods, scale)
            scale_weighted = (scale_weights[:, None] * run.preds).sum(axis=0)
            row_frame[f"pf_{spec.name}_scale_{format_scale(scale)}"] = scale_weighted.astype(
                np.float32
            )
        row_frame[f"pf_{spec.name}_lik_mean"] = primary_weighted.astype(np.float32)
        row_frame[f"pf_{spec.name}_seed_mean"] = run.preds.mean(axis=0).astype(np.float32)
        row_frame[f"pf_{spec.name}_best_seed"] = run.preds[best_idx].astype(np.float32)
        row_frame[f"pf_{spec.name}_top{top_k}_oracle"] = oracle
        row_frame[f"pf_{spec.name}_ess_mean_diag"] = run.ess_mean_by_row.astype(np.float32)
        row_frame[f"pf_{spec.name}_resampled_rate_diag"] = run.resampled_by_row.astype(np.float32)
        diagnostics.append(
            {
                "well": holdout.well,
                "filter": spec.name,
                "filter_kind": spec.kind,
                "transition": spec.transition,
                "filter_params": json.dumps(to_jsonable(spec.params), sort_keys=True),
                "transition_params": json.dumps(
                    to_jsonable(spec.transition_params),
                    sort_keys=True,
                ),
                "seed_count": int(run.preds.shape[0]),
                "rows": int(run.preds.shape[1]),
                "gr_sigma": float(run.sigma),
                "log_likelihood_mean": float(np.mean(run.log_likelihoods)),
                "log_likelihood_std": float(np.std(run.log_likelihoods)),
                "ess_mean": float(np.mean(run.ess_mean_by_row)),
                "resampling_rate": float(np.mean(run.resampled_by_row)),
                "primary_scale": primary_scale,
                "likelihood_scales": json.dumps(scales),
                "seed_weight_max": float(np.max(primary_weights)),
                "affine_fallback": bool(
                    run.filter_metadata.get("horizontal_filter", {}).get("fallback", False)
                ),
                "affine_fallback_reason": run.filter_metadata.get("horizontal_filter", {}).get(
                    "fallback_reason"
                ),
                "affine_slope": run.filter_metadata.get("horizontal_filter", {}).get("slope"),
                "affine_intercept": run.filter_metadata.get("horizontal_filter", {}).get(
                    "intercept"
                ),
                "affine_prefix_rmse": run.filter_metadata.get("horizontal_filter", {}).get(
                    "prefix_rmse"
                ),
                "affine_prefix_points": run.filter_metadata.get("horizontal_filter", {}).get(
                    "prefix_points"
                ),
                "affine_used_points": run.filter_metadata.get("horizontal_filter", {}).get(
                    "used_points"
                ),
                "structural_active": bool(
                    run.filter_metadata.get("structural_prior", {}).get("active", False)
                ),
                "structural_sigma": run.filter_metadata.get("structural_prior", {}).get("sigma"),
                "structural_weight": run.filter_metadata.get("structural_prior", {}).get(
                    "weight"
                ),
                "structural_fallback_reason": run.filter_metadata.get(
                    "structural_prior",
                    {},
                ).get("fallback_reason"),
                "filter_metadata": json.dumps(to_jsonable(run.filter_metadata), sort_keys=True),
            }
        )
    for spec in filters:
        row_frame[f"beam_{spec.name}_top1"] = beam_outputs[spec.name].astype(np.float32)

    filter_candidate_cols = []
    nonraw_candidate_cols = []
    for spec in filters:
        cols = [
            f"pf_{spec.name}_lik_mean",
            f"pf_{spec.name}_seed_mean",
            f"pf_{spec.name}_best_seed",
            f"beam_{spec.name}_top1",
        ]
        cols.extend(f"pf_{spec.name}_scale_{format_scale(scale)}" for scale in scales)
        filter_candidate_cols.extend(cols)
        if spec.name != "raw":
            nonraw_candidate_cols.extend(cols)
    rowwise_oracle(row_frame, filter_candidate_cols, "oracle_best_variant_candidate")
    rowwise_oracle(row_frame, nonraw_candidate_cols, "oracle_best_nonraw_variant_candidate")
    return row_frame, diagnostics


def selector_headroom_summary(candidate_metrics: pd.DataFrame, primary: str) -> dict[str, Any]:
    rows = candidate_metrics.set_index("candidate")
    primary_rmse = rows.loc[primary, "rmse"] if primary in rows.index else np.nan
    out: dict[str, Any] = {"primary": primary, "primary_rmse": float(primary_rmse)}
    for candidate in ["oracle_best_variant_candidate", "oracle_best_nonraw_variant_candidate"]:
        if candidate in rows.index:
            rmse = rows.loc[candidate, "rmse"]
            out[candidate] = {
                "rmse": float(rmse),
                "delta_rmse_vs_primary": float(rmse - primary_rmse)
                if np.isfinite(primary_rmse) and np.isfinite(rmse)
                else None,
            }
    return out


def run_public_raw_gr_residual_scale_control(
    config: dict[str, Any] | None = None,
    paths: ExperimentPaths | None = None,
) -> dict[str, Any]:
    started = time.time()
    config = load_config() if config is None else config
    paths = ExperimentPaths() if paths is None else paths
    paths.require_kaggle_runtime()
    paths.ensure_output_dirs()

    train_dir = paths.train_data_dir
    filters = parse_filter_specs(config)
    validation_frame, validation_meta = read_exp072_eval_cache(config)
    target_wells = select_target_wells(validation_frame, train_dir, config)
    target_well_set = set(target_wells["well"].astype(str).tolist())
    validation_frame = validation_frame[validation_frame["well"].isin(target_well_set)].copy()
    validation_rows_by_well = {
        str(well): group.copy() for well, group in validation_frame.groupby("well", sort=False)
    }

    row_frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []

    for _, target_row in target_wells.iterrows():
        well = str(target_row["well"])
        holdout = build_eval_zone_for_well(
            well,
            validation_rows_by_well.get(well, pd.DataFrame()),
            train_dir,
            config,
        )
        if holdout is None:
            status_rows.append({"well": well, "status": "skipped_no_valid_exp072_eval_zone"})
            continue
        pf_outputs: dict[str, PfRun] = {}
        beam_outputs: dict[str, np.ndarray] = {}
        for spec in filters:
            pf_outputs[spec.name] = run_pf_for_holdout(holdout, spec, config)
            beam_outputs[spec.name] = beam_search_for_holdout(holdout, spec, config)
        frame, diag = build_row_frame_for_holdout(
            holdout,
            filters,
            pf_outputs,
            beam_outputs,
            config,
        )
        row_frames.append(frame)
        diagnostics.extend(diag)
        status = dict(holdout.status)
        status.update(
            {
                "filters": ",".join(spec.name for spec in filters),
                "filter_count": int(len(filters)),
            }
        )
        status_rows.append(status)

    if not row_frames:
        raise RuntimeError("No public raw GR residual-scale control rows were generated.")

    row_frame = pd.concat(row_frames, ignore_index=True)
    pf_diagnostics = pd.DataFrame(diagnostics)
    well_status = pd.DataFrame(status_rows)
    candidate_metrics = compute_candidate_metrics(row_frame, config)
    filter_delta_metrics = compute_filter_delta_metrics(row_frame, filters)
    bucket_metrics = compute_bucket_metrics(row_frame)
    by_well = compute_by_well(row_frame, config)
    group_metrics = compute_group_metrics(row_frame)
    add_worst_well_regression(candidate_metrics, by_well)

    artifacts = paths.artifacts_dir
    candidate_metrics_path = artifacts / f"{OUTPUT_PREFIX}_candidate_metrics.csv"
    filter_delta_path = artifacts / f"{OUTPUT_PREFIX}_filter_delta_metrics.csv"
    bucket_metrics_path = artifacts / f"{OUTPUT_PREFIX}_bucket_metrics.csv"
    by_well_path = artifacts / f"{OUTPUT_PREFIX}_by_well.csv"
    group_metrics_path = artifacts / f"{OUTPUT_PREFIX}_group_metrics.csv"
    pf_diagnostics_path = artifacts / f"{OUTPUT_PREFIX}_pf_diagnostics.csv"
    target_wells_path = artifacts / f"{OUTPUT_PREFIX}_target_wells.csv"
    well_status_path = artifacts / f"{OUTPUT_PREFIX}_well_status.csv"
    row_candidates_path = artifacts / f"{OUTPUT_PREFIX}_row_candidates.csv.gz"
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"

    candidate_metrics.to_csv(candidate_metrics_path, index=False)
    filter_delta_metrics.to_csv(filter_delta_path, index=False)
    bucket_metrics.to_csv(bucket_metrics_path, index=False)
    by_well.to_csv(by_well_path, index=False)
    group_metrics.to_csv(group_metrics_path, index=False)
    pf_diagnostics.to_csv(pf_diagnostics_path, index=False)
    target_wells.to_csv(target_wells_path, index=False)
    well_status.to_csv(well_status_path, index=False)
    row_frame.to_csv(row_candidates_path, index=False, compression="gzip")

    best_row = (
        candidate_metrics.sort_values(["rmse", "candidate"], na_position="last").iloc[0].to_dict()
    )
    deployable_metrics = candidate_metrics[~candidate_metrics["is_oracle_diagnostic"]]
    best_non_oracle_row = (
        deployable_metrics.sort_values(["rmse", "candidate"], na_position="last").iloc[0].to_dict()
    )
    best_delta_vs_primary = (
        filter_delta_metrics[~filter_delta_metrics["candidate"].str.contains("_oracle")]
        .sort_values(["delta_rmse_vs_raw_family", "candidate"], na_position="last")
        .head(1)
        .to_dict("records")
    )
    primary = str(get_nested(config, "audit.primary_baseline") or "pf_raw_lik_mean")
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "completed_train_side_audit",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "runtime_seconds": float(time.time() - started),
        "rows": int(len(row_frame)),
        "wells": int(row_frame["well"].nunique()),
        "target_wells_selected": int(len(target_wells)),
        "filters": [to_jsonable(spec.__dict__) for spec in filters],
        "primary_baseline": primary,
        "best_candidate_by_rmse": best_row,
        "best_non_oracle_candidate_by_rmse": best_non_oracle_row,
        "best_delta_vs_primary": best_delta_vs_primary[0] if best_delta_vs_primary else None,
        "selector_headroom": selector_headroom_summary(candidate_metrics, primary),
        "validation_surface": validation_meta,
        "pf_diagnostics_summary": (
            pf_diagnostics.groupby("filter", observed=True)
            .agg(
                wells=("well", "nunique"),
                rows=("rows", "sum"),
                ess_mean=("ess_mean", "mean"),
                resampling_rate=("resampling_rate", "mean"),
                log_likelihood_mean=("log_likelihood_mean", "mean"),
                gr_sigma=("gr_sigma", "mean"),
            )
            .reset_index()
            .to_dict("records")
        ),
        "artifacts": {
            "candidate_metrics": str(candidate_metrics_path),
            "filter_delta_metrics": str(filter_delta_path),
            "bucket_metrics": str(bucket_metrics_path),
            "by_well": str(by_well_path),
            "group_metrics": str(group_metrics_path),
            "pf_diagnostics": str(pf_diagnostics_path),
            "target_wells": str(target_wells_path),
            "well_status": str(well_status_path),
            "row_candidates": str(row_candidates_path),
            "summary": str(summary_path),
        },
    }
    write_json(summary_path, summary)

    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": "completed_train_side_audit",
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "metric": "rmse",
        "rows": int(len(row_frame)),
        "wells": int(row_frame["well"].nunique()),
        "validation_surface": validation_meta,
        "best_candidate_by_rmse": best_row,
        "best_non_oracle_candidate_by_rmse": best_non_oracle_row,
        "best_delta_vs_primary": summary["best_delta_vs_primary"],
        "selector_headroom": summary["selector_headroom"],
        "artifacts": summary["artifacts"],
        "sha256": {
            "candidate_metrics": sha256_path(candidate_metrics_path),
            "filter_delta_metrics": sha256_path(filter_delta_path),
            "bucket_metrics": sha256_path(bucket_metrics_path),
            "by_well": sha256_path(by_well_path),
            "group_metrics": sha256_path(group_metrics_path),
            "pf_diagnostics": sha256_path(pf_diagnostics_path),
            "target_wells": sha256_path(target_wells_path),
            "well_status": sha256_path(well_status_path),
            "row_candidates_raw_gzip": sha256_path(row_candidates_path),
            "row_candidates_decompressed": sha256_path(row_candidates_path, decompressed=True),
            "summary": sha256_path(summary_path),
        },
        "notes": (
            "Train-side exp072-aligned public-like raw GR residual-scale control only; "
            "no model, inference, or submission."
        ),
    }
    write_json(paths.metrics_path, metrics)
    return {
        "summary": summary,
        "candidate_metrics": candidate_metrics,
        "filter_delta_metrics": filter_delta_metrics,
        "bucket_metrics": bucket_metrics,
        "by_well": by_well,
        "group_metrics": group_metrics,
        "pf_diagnostics": pf_diagnostics,
        "target_wells": target_wells,
        "well_status": well_status,
        "row_frame": row_frame,
    }


def outlier_mixture_variants(config: dict[str, Any]) -> list[OutlierMixtureSpec]:
    mixture = get_nested(config, "model.outlier_mixture") or {}
    component = str(mixture.get("component") or "")
    if component != "raw_gr_uniform_state_neutral":
        raise ValueError(
            "model.outlier_mixture.component must be raw_gr_uniform_state_neutral; "
            "a broad Gaussian component is not state-neutral."
        )
    uniform_gr_min = float(mixture.get("uniform_gr_min", np.nan))
    uniform_gr_max = float(mixture.get("uniform_gr_max", np.nan))
    if not (
        np.isfinite(uniform_gr_min)
        and np.isfinite(uniform_gr_max)
        and uniform_gr_min < uniform_gr_max
    ):
        raise ValueError("outlier mixture Uniform GR support must be finite and increasing")
    variants = [
        OutlierMixtureSpec(
            name=str(item["name"]),
            epsilon=float(item["epsilon"]),
            component=component,
            uniform_gr_min=uniform_gr_min,
            uniform_gr_max=uniform_gr_max,
        )
        for item in (mixture.get("variants") or [])
    ]
    if not variants:
        raise ValueError("model.outlier_mixture.variants must define at least one epsilon")
    names = [spec.name for spec in variants]
    epsilons = [spec.epsilon for spec in variants]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate outlier mixture variant names: {names}")
    if len(epsilons) != len(set(epsilons)):
        raise ValueError(f"duplicate outlier mixture epsilons: {epsilons}")
    if any(not np.isfinite(epsilon) or epsilon <= 0.0 or epsilon >= 1.0 for epsilon in epsilons):
        raise ValueError("outlier mixture epsilon must be finite and strictly between 0 and 1")

    active_names = get_nested(config, "execution.active_outlier_mixture_variants")
    if active_names is None:
        return variants
    if not isinstance(active_names, list) or not active_names:
        raise ValueError(
            "execution.active_outlier_mixture_variants must be a non-empty list or null"
        )
    active_names = [str(name) for name in active_names]
    if len(active_names) != len(set(active_names)):
        raise ValueError("execution.active_outlier_mixture_variants must not contain duplicates")
    known_names = {spec.name for spec in variants}
    unknown_names = sorted(set(active_names) - known_names)
    if unknown_names:
        raise ValueError(
            "execution.active_outlier_mixture_variants contains unknown mixture variants: "
            f"{unknown_names}"
        )
    selected = [spec for spec in variants if spec.name in set(active_names)]
    if len(selected) != len(active_names):
        raise RuntimeError("failed to resolve every active outlier-mixture variant")
    return selected


def config_for_single_outlier_mixture_variant(
    config: dict[str, Any],
    variant_name: str,
) -> dict[str, Any]:
    """Return an independent full-surface execution config for one declared variant.

    The base experiment configuration remains the source of the two fixed mixture
    definitions.  Variant notebooks use this helper so concurrently prepared Kaggle
    packages cannot mutate each other's in-memory selection or depend on a manual
    config edit between pushes.
    """

    base_variants = outlier_mixture_variants(config)
    if variant_name not in {spec.name for spec in base_variants}:
        raise ValueError(f"unknown declared outlier-mixture variant: {variant_name}")
    selected_config = deepcopy(config)
    execution = selected_config.setdefault("execution", {})
    if not isinstance(execution, dict):
        raise ValueError("execution must be a mapping when selecting a mixture variant")
    execution["active_outlier_mixture_variants"] = [str(variant_name)]
    selected_variants = outlier_mixture_variants(selected_config)
    if [spec.name for spec in selected_variants] != [str(variant_name)]:
        raise RuntimeError("single-variant execution selection did not resolve exactly once")
    return selected_config


def interval_well_set(target_wells: pd.DataFrame, config: dict[str, Any]) -> set[str]:
    interval_audit = get_nested(config, "model.interval_audit") or {}
    if not bool(interval_audit.get("enabled", True)):
        return set()
    count = int(interval_audit.get("selected_well_count", 64))
    keyed = [
        (stable_seed("interval_audit", str(well)), str(well))
        for well in target_wells["well"].astype(str)
    ]
    return {well for _, well in sorted(keyed)[:count]}


def aggregate_seed_interval(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    counts = finite.sum(axis=0)
    sums = np.where(finite, values, 0.0).sum(axis=0)
    out = np.full(values.shape[1], np.nan, dtype=np.float32)
    valid = counts > 0
    out[valid] = (sums[valid] / counts[valid]).astype(np.float32)
    return out


def build_mixture_row_frame(
    holdout: PrefixHoldout,
    outputs: dict[str, PfRun],
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    eval_rows = holdout.masked.loc[holdout.eval_index]
    md_since = (
        holdout.cache_md_since
        if holdout.cache_md_since is not None
        else numeric_array(eval_rows, "MD") - np.float32(holdout.last_known_md)
    )
    row_frame = pd.DataFrame(
        {
            "well": holdout.well,
            "row_idx": holdout.eval_index.astype(np.int32),
            "id": holdout.eval_ids.astype(str),
            "target": holdout.target_delta.astype(np.float32),
            "true_tvt": holdout.true_tvt.astype(np.float32),
            "last_known_tvt": np.float32(holdout.last_known_tvt),
            "last_known_md": np.float32(holdout.last_known_md),
            "md_since": md_since.astype(np.float32),
        }
    )
    for column in holdout.reference_candidates.columns:
        row_frame[f"exp072_{column}"] = pd.to_numeric(
            holdout.reference_candidates[column], errors="coerce",
        ).to_numpy(np.float32)

    pf_diagnostics: list[dict[str, Any]] = []
    gate_diagnostics: list[dict[str, Any]] = []
    for variant_name, run in outputs.items():
        prediction = run.preds.mean(axis=0).astype(np.float32)
        interval_p05 = aggregate_seed_interval(run.interval_p05_by_seed)
        interval_p95 = aggregate_seed_interval(run.interval_p95_by_seed)
        row_frame[f"pf_{variant_name}_mean"] = prediction
        row_frame[f"pf_{variant_name}_p05_diag"] = interval_p05
        row_frame[f"pf_{variant_name}_p95_diag"] = interval_p95
        row_frame[f"pf_{variant_name}_ess_diag"] = run.ess_mean_by_row
        row_frame[f"pf_{variant_name}_resampled_diag"] = run.resampled_by_row
        row_frame[f"pf_{variant_name}_gate_rate_diag"] = run.gate_rate_by_row
        row_frame[f"pf_{variant_name}_mixture_rate_diag"] = run.mixture_rate_by_row
        row_frame[f"pf_{variant_name}_innovation_diag"] = run.innovation_by_row
        row_frame[f"pf_{variant_name}_change_point_diag"] = run.change_point_by_row
        row_frame[f"pf_{variant_name}_novelty_diag"] = run.novelty_by_row
        row_frame[f"pf_{variant_name}_preupdate_ess_ratio_diag"] = run.preupdate_ess_ratio_by_row
        row_frame[f"pf_{variant_name}_preupdate_max_weight_diag"] = run.preupdate_max_weight_by_row
        mixture = run.filter_metadata["public_like"]["mixture"]
        if not np.array_equal(run.gate_rate_by_row, run.mixture_rate_by_row):
            raise RuntimeError(f"{variant_name}: mixture rate differs from gate rate")
        pf_diagnostics.append(
            {
                "well": holdout.well,
                "variant": variant_name,
                "component": mixture["component"],
                "epsilon": mixture["epsilon"],
                "uniform_gr_min": mixture["uniform_gr_min"],
                "uniform_gr_max": mixture["uniform_gr_max"],
                "uniform_likelihood": mixture["uniform_likelihood"],
                "seed_count": int(run.preds.shape[0]),
                "rows": int(run.preds.shape[1]),
                "gr_sigma": float(run.sigma),
                "log_likelihood_mean": float(np.mean(run.log_likelihoods)),
                "log_likelihood_std": float(np.std(run.log_likelihoods)),
                "ess_mean": float(np.mean(run.ess_mean_by_row)),
                "resampling_rate": float(np.mean(run.resampled_by_row)),
                "record_intervals": bool(run.filter_metadata["public_like"]["record_intervals"]),
            }
        )
        gate_diagnostics.append(
            {
                "well": holdout.well,
                "variant": variant_name,
                "epsilon": mixture["epsilon"],
                "rows": int(len(row_frame)),
                "gate_seed_fraction_mean": float(np.mean(run.gate_rate_by_row)),
                "mixture_seed_fraction_mean": float(np.mean(run.mixture_rate_by_row)),
                "gate_any_seed_rows": int(np.sum(run.gate_rate_by_row > 0.0)),
                "gate_all_seed_rows": int(np.sum(run.gate_rate_by_row >= 1.0)),
                "mixture_any_seed_rows": int(np.sum(run.mixture_rate_by_row > 0.0)),
                "mixture_all_seed_rows": int(np.sum(run.mixture_rate_by_row >= 1.0)),
                "innovation_mean": float(np.mean(run.innovation_by_row)),
                "change_point_mean": float(np.mean(run.change_point_by_row)),
                "novelty_mean": float(np.mean(run.novelty_by_row)),
                "preupdate_ess_ratio_mean": float(np.mean(run.preupdate_ess_ratio_by_row)),
                "preupdate_max_weight_mean": float(np.mean(run.preupdate_max_weight_by_row)),
                "non_gate_likelihood": "exp072_gaussian_exact",
            }
        )
    return row_frame, pf_diagnostics, gate_diagnostics


def resolve_existing_candidate(paths: ExperimentPaths, candidates: list[str]) -> Path | None:
    for raw in candidates:
        candidate = Path(raw)
        for path in [candidate, paths.root / candidate, Path.cwd() / candidate]:
            if path.exists() and path.stat().st_size > 0:
                return path
    if KAGGLE_INPUT_ROOT.exists() and candidates:
        filename = Path(candidates[0]).name
        for path in KAGGLE_INPUT_ROOT.glob(f"**/{filename}"):
            if path.exists() and path.stat().st_size > 0:
                return path
    return None


def load_temperature_reference(
    paths: ExperimentPaths,
    config: dict[str, Any],
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    comparison = get_nested(config, "comparison.temperature_experiment") or {}
    if not bool(comparison.get("enabled", True)):
        raise ValueError("temperature comparison must remain enabled for exp233")
    row_candidates_name = str(comparison.get("row_candidates_filename") or "")
    if not row_candidates_name:
        raise ValueError("temperature comparison row_candidates_filename is required")
    row_path = find_existing_path(
        filename=row_candidates_name,
        candidates=list(comparison.get("row_candidates_candidates") or []),
    )
    if row_path is None:
        if bool(comparison.get("allow_pending_during_parallel_train", False)):
            return None, {
                "status": "pending_exp232_artifacts",
                "required_for_acceptance": bool(comparison.get("required_for_acceptance", True)),
                "row_candidates_filename": row_candidates_name,
            }
        raise FileNotFoundError(
            "exp232 temperature row candidates are required before this mixture audit can run"
        )

    required_artifacts = [str(name) for name in comparison.get("required_artifacts") or []]
    missing_artifacts = [
        filename
        for filename in required_artifacts
        if find_existing_path(filename=filename) is None
    ]
    if missing_artifacts:
        raise FileNotFoundError(
            "exp232 temperature comparison is incomplete; missing artifacts: "
            f"{missing_artifacts}"
        )

    temperature = pd.read_csv(row_path, compression="infer")
    expected_columns = [
        "id",
        "well",
        "row_idx",
        *[str(name) for name in comparison.get("required_candidate_columns") or []],
    ]
    missing_columns = [column for column in expected_columns if column not in temperature.columns]
    if missing_columns:
        raise ValueError(
            "exp232 temperature row candidates are missing expected columns: "
            f"{missing_columns}"
        )
    if temperature["id"].astype(str).duplicated().any():
        raise ValueError("exp232 temperature row candidates contain duplicate ids")
    columns = [
        "id",
        "well",
        "row_idx",
        *[
            column
            for column in temperature.columns
            if column.startswith("pf_temp_") and (column.endswith("_mean") or column.endswith("_diag"))
        ],
    ]
    return temperature[columns].copy(), {
        "status": "available",
        "required_for_acceptance": bool(comparison.get("required_for_acceptance", True)),
        "row_candidates_path": str(row_path),
        "row_candidates_rows": int(len(temperature)),
        "required_artifacts": required_artifacts,
    }


def attach_temperature_reference(
    row_frame: pd.DataFrame,
    temperature: pd.DataFrame,
) -> pd.DataFrame:
    if row_frame["id"].astype(str).duplicated().any():
        raise ValueError("mixture row candidates contain duplicate ids")
    comparison = temperature.rename(
        columns={"well": "temperature_well", "row_idx": "temperature_row_idx"}
    )
    joined = row_frame.merge(comparison, how="left", on="id", validate="one_to_one")
    if joined["temperature_well"].isna().any():
        examples = joined.loc[joined["temperature_well"].isna(), "id"].head(5).tolist()
        raise ValueError(f"exp232 temperature rows are missing mixture ids, examples={examples}")
    if not np.array_equal(
        joined["well"].astype(str).to_numpy(), joined["temperature_well"].astype(str).to_numpy()
    ):
        raise ValueError("exp232 temperature row well ids do not match the mixture evaluation surface")
    if not np.array_equal(
        joined["row_idx"].to_numpy(), joined["temperature_row_idx"].to_numpy()
    ):
        raise ValueError("exp232 temperature row indices do not match the mixture evaluation surface")
    return joined.drop(columns=["temperature_well", "temperature_row_idx"])


def compute_hidden_like_metrics(
    frame: pd.DataFrame,
    paths: ExperimentPaths,
    config: dict[str, Any],
) -> pd.DataFrame:
    hidden = get_nested(config, "comparison.hidden_like") or {}
    if not bool(hidden.get("enabled", False)):
        return pd.DataFrame()
    path = resolve_existing_candidate(paths, list(hidden.get("fold_assignment_candidates") or []))
    if path is None:
        return pd.DataFrame()
    assignments = pd.read_csv(path, dtype={"well_id": str})
    true = numeric_array(frame, "true_tvt")
    rows: list[dict[str, Any]] = []
    for subgroup, role_column in (hidden.get("valid_role_columns") or {}).items():
        if role_column not in assignments.columns:
            continue
        wells = set(assignments.loc[assignments[role_column].astype(str) == "valid", "well_id"].astype(str))
        mask = frame["well"].astype(str).isin(wells).to_numpy()
        if not mask.any():
            continue
        for candidate in candidate_columns(frame):
            rows.append(
                {
                    "subgroup": str(subgroup),
                    "candidate": candidate,
                    **score_prediction(numeric_array(frame, candidate)[mask], true[mask]),
                }
            )
    return pd.DataFrame(rows)


def compute_interval_metrics(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    true = numeric_array(frame, "true_tvt")
    metric_rows: list[dict[str, Any]] = []
    loss_rows: list[dict[str, Any]] = []
    for mean_column in [column for column in frame.columns if column.startswith("pf_") and column.endswith("_mean")]:
        prefix = mean_column.removesuffix("_mean")
        lower_column = f"{prefix}_p05_diag"
        upper_column = f"{prefix}_p95_diag"
        gate_column = f"{prefix}_gate_rate_diag"
        if lower_column not in frame or upper_column not in frame or gate_column not in frame:
            continue
        lower = numeric_array(frame, lower_column)
        upper = numeric_array(frame, upper_column)
        gate_rate = numeric_array(frame, gate_column)
        sampled = np.isfinite(lower) & np.isfinite(upper) & np.isfinite(true)
        coverage = (true >= lower) & (true <= upper)
        interval_slices = {
            "all_sampled": sampled,
            "gate_any_seed_sampled": sampled & (gate_rate > 0.0),
            "gate_all_seed_sampled": sampled & (gate_rate >= 1.0),
        }
        for slice_name, slice_mask in interval_slices.items():
            metric_rows.append(
                {
                    "candidate": mean_column,
                    "slice": slice_name,
                    "rows": int(slice_mask.sum()),
                    "coverage": float(np.mean(coverage[slice_mask])) if slice_mask.any() else None,
                    "mean_width": (
                        float(np.mean((upper - lower)[slice_mask])) if slice_mask.any() else None
                    ),
                }
            )
        for bucket in pd.Series(distance_bucket(frame["md_since"])).cat.categories:
            bucket_mask = np.asarray(distance_bucket(frame["md_since"]) == bucket, dtype=bool) & sampled
            if bucket_mask.any():
                metric_rows.append(
                    {
                        "candidate": mean_column,
                        "slice": str(bucket),
                        "rows": int(bucket_mask.sum()),
                        "coverage": float(np.mean(coverage[bucket_mask])),
                        "mean_width": float(np.mean((upper - lower)[bucket_mask])),
                    }
                )
        work = frame[["well", "row_idx", "md_since"]].copy()
        work["sampled"] = sampled
        work["covered"] = coverage
        for well, group in work.sort_values(["well", "row_idx"]).groupby("well", sort=False):
            losses = group[group["sampled"] & ~group["covered"]]
            loss_rows.append(
                {
                    "candidate": mean_column,
                    "well": str(well),
                    "sampled_rows": int(group["sampled"].sum()),
                    "first_sampled_loss_row_idx": int(losses.iloc[0]["row_idx"]) if len(losses) else None,
                    "first_sampled_loss_md_since": float(losses.iloc[0]["md_since"]) if len(losses) else None,
                    "ever_lost": bool(len(losses)),
                }
            )
    return pd.DataFrame(metric_rows), pd.DataFrame(loss_rows)


def run_adaptive_outlier_mixture_likelihood_pf(
    config: dict[str, Any] | None = None,
    paths: ExperimentPaths | None = None,
) -> dict[str, Any]:
    started = time.time()
    config = load_config() if config is None else config
    paths = ExperimentPaths() if paths is None else paths
    paths.require_kaggle_runtime()
    paths.ensure_output_dirs()

    validation_frame, validation_meta = read_exp072_eval_cache(config)
    target_wells = select_target_wells(validation_frame, paths.train_data_dir, config)
    if target_wells.empty:
        raise RuntimeError("No eligible exp072 pseudo-tail wells were selected.")
    variants = outlier_mixture_variants(config)
    temperature_reference, temperature_comparison = load_temperature_reference(paths, config)
    interval_wells = interval_well_set(target_wells, config)
    validation_rows_by_well = {
        str(well): group.copy() for well, group in validation_frame.groupby("well", sort=False)
    }
    row_frames: list[pd.DataFrame] = []
    pf_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    for _, target_row in target_wells.iterrows():
        well = str(target_row["well"])
        holdout = build_eval_zone_for_well(
            well,
            validation_rows_by_well.get(well, pd.DataFrame()),
            paths.train_data_dir,
            config,
        )
        if holdout is None:
            status_rows.append({"well": well, "status": "skipped_no_valid_exp072_eval_zone"})
            continue
        outputs = {
            mixture.name: run_pf_for_holdout(
                holdout,
                mixture=mixture,
                record_intervals=well in interval_wells,
                config=config,
            )
            for mixture in variants
        }
        row_frame, pf_diag, gate_diag = build_mixture_row_frame(holdout, outputs)
        row_frames.append(row_frame)
        pf_rows.extend(pf_diag)
        gate_rows.extend(gate_diag)
        status = dict(holdout.status)
        status.update({"interval_audit_selected": well in interval_wells, "status": "ok"})
        status_rows.append(status)
    if not row_frames:
        raise RuntimeError("No adaptive likelihood-PF rows were generated.")

    row_frame = pd.concat(row_frames, ignore_index=True)
    if temperature_reference is not None:
        row_frame = attach_temperature_reference(row_frame, temperature_reference)
    pf_diagnostics = pd.DataFrame(pf_rows)
    gate_diagnostics = pd.DataFrame(gate_rows)
    well_status = pd.DataFrame(status_rows)
    candidate_metrics = compute_candidate_metrics(row_frame, config)
    bucket_metrics = compute_bucket_metrics(row_frame)
    by_well = compute_by_well(row_frame, config)
    add_worst_well_regression(candidate_metrics, by_well)
    hidden_like_metrics = compute_hidden_like_metrics(row_frame, paths, config)
    interval_metrics, first_loss_by_well = compute_interval_metrics(row_frame)

    artifacts = paths.artifacts_dir
    artifact_paths = {
        "candidate_metrics": artifacts / f"{OUTPUT_PREFIX}_candidate_metrics.csv",
        "distance_bucket_metrics": artifacts / f"{OUTPUT_PREFIX}_distance_bucket_metrics.csv",
        "hidden_like_metrics": artifacts / f"{OUTPUT_PREFIX}_hidden_like_metrics.csv",
        "by_well": artifacts / f"{OUTPUT_PREFIX}_by_well.csv",
        "pf_diagnostics": artifacts / f"{OUTPUT_PREFIX}_pf_diagnostics.csv",
        "gate_diagnostics": artifacts / f"{OUTPUT_PREFIX}_gate_diagnostics.csv",
        "interval_metrics": artifacts / f"{OUTPUT_PREFIX}_interval_metrics.csv",
        "first_sampled_loss_by_well": artifacts / f"{OUTPUT_PREFIX}_first_sampled_loss_by_well.csv",
        "target_wells": artifacts / f"{OUTPUT_PREFIX}_target_wells.csv",
        "well_status": artifacts / f"{OUTPUT_PREFIX}_well_status.csv",
        "row_candidates": artifacts / f"{OUTPUT_PREFIX}_row_candidates.csv.gz",
        "summary": artifacts / f"{OUTPUT_PREFIX}_summary.json",
    }
    candidate_metrics.to_csv(artifact_paths["candidate_metrics"], index=False)
    bucket_metrics.to_csv(artifact_paths["distance_bucket_metrics"], index=False)
    hidden_like_metrics.to_csv(artifact_paths["hidden_like_metrics"], index=False)
    by_well.to_csv(artifact_paths["by_well"], index=False)
    pf_diagnostics.to_csv(artifact_paths["pf_diagnostics"], index=False)
    gate_diagnostics.to_csv(artifact_paths["gate_diagnostics"], index=False)
    interval_metrics.to_csv(artifact_paths["interval_metrics"], index=False)
    first_loss_by_well.to_csv(artifact_paths["first_sampled_loss_by_well"], index=False)
    target_wells.to_csv(artifact_paths["target_wells"], index=False)
    well_status.to_csv(artifact_paths["well_status"], index=False)
    row_frame.to_csv(artifact_paths["row_candidates"], index=False, compression="gzip")

    primary = str(get_nested(config, "audit.primary_baseline") or "exp072_likpf_mean")
    primary_row = candidate_metrics.loc[candidate_metrics["candidate"] == primary]
    best_new = candidate_metrics[
        candidate_metrics["candidate"].str.startswith("pf_mix_")
    ].sort_values(["rmse", "candidate"], na_position="last")
    execution = get_nested(config, "execution") or {}
    execution_summary = {
        "split_strategy": execution.get("split_strategy", "all_variants_single_kernel"),
        "active_outlier_mixture_variants": [mixture.name for mixture in variants],
        "full_eligible_well_surface": bool(
            get_nested(config, "model.validation_surface.max_target_wells") is None
        ),
    }
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "completed_train_side_audit",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "runtime_seconds": float(time.time() - started),
        "rows": int(len(row_frame)),
        "wells": int(row_frame["well"].nunique()),
        "target_wells_selected": int(len(target_wells)),
        "interval_audit_wells": int(len(interval_wells)),
        "execution": execution_summary,
        "control": {"candidate": primary, "regenerated": False},
        "outlier_mixture_variants": [
            {
                "name": mixture.name,
                "epsilon": mixture.epsilon,
                "component": mixture.component,
                "uniform_gr_min": mixture.uniform_gr_min,
                "uniform_gr_max": mixture.uniform_gr_max,
            }
            for mixture in variants
        ],
        "primary_baseline_metrics": primary_row.to_dict("records")[0] if len(primary_row) else None,
        "best_outlier_mixture_variant": best_new.iloc[0].to_dict() if len(best_new) else None,
        "hidden_like_metrics_available": bool(len(hidden_like_metrics)),
        "interval_coverage": interval_metrics.to_dict("records"),
        "temperature_comparison": temperature_comparison,
        "gate_summary": (
            gate_diagnostics.groupby("variant", observed=True)
            .agg(
                wells=("well", "nunique"),
                rows=("rows", "sum"),
                gate_seed_fraction_mean=("gate_seed_fraction_mean", "mean"),
                gate_any_seed_rows=("gate_any_seed_rows", "sum"),
                gate_all_seed_rows=("gate_all_seed_rows", "sum"),
            )
            .reset_index()
            .to_dict("records")
        ),
        "validation_surface": validation_meta,
        "artifacts": {key: str(path) for key, path in artifact_paths.items()},
    }
    write_json(artifact_paths["summary"], summary)
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": "completed_train_side_audit",
        "route": "pf_beam",
        "metric": "rmse",
        "cv": None,
        "public_lb": None,
        "private_lb": None,
        "rows": summary["rows"],
        "wells": summary["wells"],
        "execution": summary["execution"],
        "control": summary["control"],
        "outlier_mixture_variants": summary["outlier_mixture_variants"],
        "primary_baseline_metrics": summary["primary_baseline_metrics"],
        "best_outlier_mixture_variant": summary["best_outlier_mixture_variant"],
        "hidden_like_metrics_available": summary["hidden_like_metrics_available"],
        "temperature_comparison": summary["temperature_comparison"],
        "gate_summary": summary["gate_summary"],
        "artifacts": summary["artifacts"],
        "sha256": {
            **{
                key: sha256_path(path)
                for key, path in artifact_paths.items()
                if key != "row_candidates"
            },
            "row_candidates_raw_gzip": sha256_path(artifact_paths["row_candidates"]),
            "row_candidates_decompressed": sha256_path(
                artifact_paths["row_candidates"],
                decompressed=True,
            ),
        },
        "notes": (
            "State-neutral Uniform outlier-mixture likelihood-PF train-side audit; "
            "no inference or submission. Temperature comparison is required for acceptance."
        ),
    }
    write_json(paths.metrics_path, metrics)
    return {
        "summary": summary,
        "candidate_metrics": candidate_metrics,
        "bucket_metrics": bucket_metrics,
        "hidden_like_metrics": hidden_like_metrics,
        "by_well": by_well,
        "pf_diagnostics": pf_diagnostics,
        "gate_diagnostics": gate_diagnostics,
        "interval_metrics": interval_metrics,
        "first_loss_by_well": first_loss_by_well,
        "target_wells": target_wells,
        "well_status": well_status,
    }


def main() -> dict[str, Any]:
    return run_adaptive_outlier_mixture_likelihood_pf()


if __name__ == "__main__":
    result = main()
    print(json.dumps(to_jsonable(result["summary"]), indent=2, sort_keys=True))
