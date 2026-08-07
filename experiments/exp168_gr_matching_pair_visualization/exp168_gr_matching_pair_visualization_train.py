# %% [markdown]
# # exp168_gr_matching_pair_visualization train

# %% [markdown]
# ## Contents
# 1. Imports
# 2. Runtime and configuration helpers
# 3. GR matching helpers
# 4. Pair selection and plotting helpers
# 5. Setup and input checks
# 6. Score GR matching pairs
# 7. Generate visualizations
# 8. Summary and generated artifacts

# %%
from __future__ import annotations

import gzip
import hashlib
import html
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from IPython.display import display

EXPERIMENT_NAME = "exp168_gr_matching_pair_visualization"
OUTPUT_PREFIX = "exp168_gr_matching_pair_visualization"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")

# %% [markdown]
# ## 2. Runtime and configuration helpers

# %%
def get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open() as fp:
        value = yaml.safe_load(fp) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def find_project_root() -> Path:
    cwd = Path.cwd()
    for candidate in (cwd, *cwd.parents):
        if (candidate / "project.yml").exists() and (candidate / "experiments").exists():
            return candidate
    return cwd


def load_config() -> dict[str, Any]:
    root = find_project_root()
    project = read_yaml(root / "project.yml")
    experiment = read_yaml(Path("config.yaml"))
    if not experiment:
        experiment = read_yaml(root / "experiments" / EXPERIMENT_NAME / "config.yaml")
    defaults = {
        "data": {
            "train_dir": get_nested(project, "data.train_dir", "data/raw/train"),
        },
        "runtime": {
            "kaggle": get_nested(project, "runtime.kaggle", {}),
        },
    }
    return deep_merge(defaults, experiment)


def is_kaggle_runtime() -> bool:
    return KAGGLE_INPUT_ROOT.exists() and KAGGLE_WORKING_ROOT.exists()


def resolve_train_dir(config: dict[str, Any]) -> Path:
    configured = Path(str(get_nested(config, "data.train_dir", "data/raw/train")))
    if configured.exists():
        return configured
    root = find_project_root()
    local = root / configured
    if local.exists():
        return local
    if KAGGLE_INPUT_ROOT.exists():
        candidates = [
            KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction" / "train",
            KAGGLE_INPUT_ROOT / "competitions" / "rogii-wellbore-geology-prediction" / "train",
        ]
        candidates.extend(sorted(KAGGLE_INPUT_ROOT.glob("**/train")))
        for candidate in candidates:
            if candidate.exists() and any(candidate.glob("*__horizontal_well.csv")):
                return candidate
    raise FileNotFoundError(f"Could not resolve train directory from {configured}")


def resolve_output_dirs() -> tuple[Path, Path]:
    if is_kaggle_runtime():
        artifact_dir = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        artifact_dir = find_project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    figures_dir = artifact_dir / "figures"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir, figures_dir


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(float(value)) else None
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except TypeError:
        pass
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_decompressed(path: Path) -> str | None:
    if path.suffix != ".gz":
        return None
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# %% [markdown]
# ## 3. GR matching helpers

# %%
@dataclass(frozen=True)
class FilteredSeries:
    name: str
    values: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class WellBundle:
    well: str
    horizontal: pd.DataFrame
    typewell: pd.DataFrame
    md: np.ndarray
    true_tvt: np.ndarray
    tvt_input: np.ndarray
    known_end: int
    full_gr: np.ndarray
    type_tvt: np.ndarray
    type_gr: np.ndarray
    filters: dict[str, FilteredSeries]


def fill_numeric(values: pd.Series | np.ndarray, fallback: float = 0.0) -> np.ndarray:
    series = pd.Series(values, dtype="float64")
    if series.notna().any():
        fallback = float(series.mean())
    filled = series.interpolate(limit_direction="both").ffill().bfill().fillna(fallback)
    return filled.to_numpy(np.float32)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values)
        .rolling(int(window), center=True, min_periods=1)
        .mean()
        .to_numpy(np.float32)
    )


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    return (
        pd.Series(values)
        .rolling(int(window), center=True, min_periods=1)
        .median()
        .to_numpy(np.float32)
    )


def savgol_or_rolling_mean(values: np.ndarray, *, window: int, polyorder: int) -> tuple[np.ndarray, dict[str, Any]]:
    window = int(window)
    if window % 2 == 0:
        window += 1
    if len(values) < window or window <= int(polyorder):
        effective = max(3, min(len(values), window))
        return rolling_mean(values, effective), {"effective_kind": "rolling_mean_short_series", "window": effective}
    try:
        from scipy.signal import savgol_filter

        filtered = savgol_filter(values, window_length=window, polyorder=int(polyorder), mode="interp")
        return filtered.astype(np.float32), {"effective_kind": "savgol", "window": window, "polyorder": int(polyorder)}
    except Exception as exc:
        return rolling_mean(values, window), {
            "effective_kind": "rolling_mean_fallback",
            "window": window,
            "polyorder": int(polyorder),
            "fallback_reason": type(exc).__name__,
        }


def fft_notch(
    values: np.ndarray,
    *,
    max_peaks: int,
    min_period_ft: float,
    max_period_ft: float,
    notch_width_bins: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = int(len(values))
    if n < 8:
        return values.astype(np.float32), {"effective_kind": "raw_short_series", "peaks": []}
    centered = values.astype(np.float64) - float(np.mean(values))
    spectrum = np.fft.rfft(centered)
    freqs = np.fft.rfftfreq(n, d=1.0)
    power = np.abs(spectrum) ** 2
    valid = (freqs >= 1.0 / max(float(max_period_ft), 1e-6)) & (
        freqs <= 1.0 / max(float(min_period_ft), 1e-6)
    )
    valid[0] = False
    candidate_idx = np.flatnonzero(valid)
    if candidate_idx.size == 0:
        return values.astype(np.float32), {"effective_kind": "raw_no_valid_fft_band", "peaks": []}
    order = candidate_idx[np.argsort(power[candidate_idx])[::-1]]
    selected: list[int] = []
    width = int(max(1, notch_width_bins))
    for idx in order:
        if all(abs(int(idx) - prev) > width for prev in selected):
            selected.append(int(idx))
        if len(selected) >= int(max_peaks):
            break
    filtered = spectrum.copy()
    for idx in selected:
        lo = max(1, idx - width)
        hi = min(len(filtered) - 1, idx + width)
        filtered[lo : hi + 1] = 0.0
    denoised = np.fft.irfft(filtered, n=n) + float(np.mean(values))
    peaks = [
        {
            "bin": int(idx),
            "frequency_cyc_per_ft": float(freqs[idx]),
            "period_ft": float(1.0 / freqs[idx]) if freqs[idx] > 0 else None,
            "power": float(power[idx]),
        }
        for idx in selected
    ]
    return denoised.astype(np.float32), {
        "effective_kind": "fft_notch",
        "max_peaks": int(max_peaks),
        "notch_width_bins": width,
        "min_period_ft": float(min_period_ft),
        "max_period_ft": float(max_period_ft),
        "peaks": peaks,
    }


def build_filters(values: np.ndarray, filter_specs: list[dict[str, Any]]) -> dict[str, FilteredSeries]:
    filters: dict[str, FilteredSeries] = {}
    for spec in filter_specs:
        name = str(spec["name"])
        kind = str(spec.get("kind", "raw"))
        if kind == "raw":
            filtered = values.astype(np.float32)
            metadata = {"effective_kind": "raw"}
        elif kind == "rolling_median":
            window = int(spec.get("window", 11))
            filtered = rolling_median(values, window)
            metadata = {"effective_kind": "rolling_median", "window": window}
        elif kind == "rolling_mean":
            window = int(spec.get("window", 21))
            filtered = rolling_mean(values, window)
            metadata = {"effective_kind": "rolling_mean", "window": window}
        elif kind == "savgol":
            filtered, metadata = savgol_or_rolling_mean(
                values,
                window=int(spec.get("window", 31)),
                polyorder=int(spec.get("polyorder", 2)),
            )
        elif kind == "fft_notch":
            filtered, metadata = fft_notch(
                values,
                max_peaks=int(spec.get("max_peaks", 2)),
                min_period_ft=float(spec.get("min_period_ft", 6.0)),
                max_period_ft=float(spec.get("max_period_ft", 120.0)),
                notch_width_bins=int(spec.get("notch_width_bins", 2)),
            )
        else:
            raise ValueError(f"Unknown filter kind: {kind}")
        filters[name] = FilteredSeries(name=name, values=filtered.astype(np.float32), metadata=metadata)
    return filters


def deterministic_eval_indices(start: int, stop: int, max_rows: int) -> np.ndarray:
    if stop <= start:
        return np.zeros(0, dtype=np.int32)
    rows = np.arange(start, stop, dtype=np.int32)
    if len(rows) <= int(max_rows):
        return rows
    positions = np.linspace(0, len(rows) - 1, int(max_rows))
    return rows[np.unique(np.rint(positions).astype(np.int32))].astype(np.int32)


def prefix_slope_prior(
    *,
    md: np.ndarray,
    tvt_input: np.ndarray,
    known_end: int,
    slope_window_rows: int,
    slope_clip: tuple[float, float],
) -> tuple[np.ndarray, dict[str, Any]]:
    if known_end <= 1:
        raise ValueError("known_end must include at least two prefix rows")
    fit_start = max(0, int(known_end) - int(slope_window_rows))
    fit_md = md[fit_start:known_end].astype(np.float64)
    fit_tvt = tvt_input[fit_start:known_end].astype(np.float64)
    finite = np.isfinite(fit_md) & np.isfinite(fit_tvt)
    if finite.sum() >= 2 and float(np.nanstd(fit_md[finite])) > 1e-6:
        slope, intercept = np.polyfit(fit_md[finite], fit_tvt[finite], deg=1)
    else:
        slope = 1.0
        intercept = float(tvt_input[known_end - 1] - md[known_end - 1])
    unclipped_slope = float(slope)
    slope = float(np.clip(slope, float(slope_clip[0]), float(slope_clip[1])))
    last_md = float(md[known_end - 1])
    last_tvt = float(tvt_input[known_end - 1])
    prior = last_tvt + slope * (md.astype(np.float64) - last_md)
    return prior.astype(np.float32), {
        "known_end": int(known_end),
        "fit_start": int(fit_start),
        "fit_rows": int(finite.sum()),
        "unclipped_slope": unclipped_slope,
        "slope": slope,
        "intercept": float(intercept),
        "last_md": last_md,
        "last_tvt": last_tvt,
    }


def standardize_rows(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=-1, keepdims=True)
    scale = values.std(axis=-1, keepdims=True) + 1e-6
    return centered / scale


def gather_horizontal(series: np.ndarray, centers: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    idx = np.clip(centers[:, None] + offsets[None, :], 0, len(series) - 1)
    return series[idx].astype(np.float32)


def interpolate_typewell(type_tvt: np.ndarray, type_gr: np.ndarray, candidate_tvt: np.ndarray) -> np.ndarray:
    flat = np.interp(
        candidate_tvt.reshape(-1),
        type_tvt.astype(np.float64),
        type_gr.astype(np.float64),
        left=float(type_gr[0]),
        right=float(type_gr[-1]),
    )
    return flat.reshape(candidate_tvt.shape).astype(np.float32)


def list_wells(train_dir: Path, audit_config: dict[str, Any]) -> list[str]:
    include = [str(value) for value in audit_config.get("well_include", []) if value]
    if include:
        wells = include
    else:
        wells = sorted(path.name.removesuffix("__horizontal_well.csv") for path in train_dir.glob("*__horizontal_well.csv"))
    max_wells = audit_config.get("max_wells")
    if max_wells is not None:
        wells = wells[: int(max_wells)]
    if not wells:
        raise FileNotFoundError(f"No horizontal well files found in {train_dir}")
    return wells


def load_well_bundle(well: str, train_dir: Path, audit_config: dict[str, Any]) -> WellBundle:
    horizontal_path = train_dir / f"{well}__horizontal_well.csv"
    typewell_path = train_dir / f"{well}__typewell.csv"
    horizontal = pd.read_csv(horizontal_path)
    typewell = pd.read_csv(typewell_path)
    required_horizontal = {"MD", "TVT", "GR", "TVT_input"}
    required_typewell = {"TVT", "GR"}
    missing_h = required_horizontal - set(horizontal.columns)
    missing_t = required_typewell - set(typewell.columns)
    if missing_h or missing_t:
        raise ValueError(f"{well} missing columns: horizontal={missing_h}, typewell={missing_t}")
    md = fill_numeric(horizontal["MD"])
    true_tvt = fill_numeric(horizontal["TVT"])
    tvt_input_raw = pd.to_numeric(horizontal["TVT_input"], errors="coerce")
    known = tvt_input_raw.notna().to_numpy()
    if not known.any():
        raise ValueError(f"No TVT_input prefix for {well}")
    known_end = int(np.flatnonzero(known)[-1] + 1)
    tvt_input = fill_numeric(tvt_input_raw)
    type_sorted = typewell[["TVT", "GR"]].dropna().sort_values("TVT")
    type_tvt = pd.to_numeric(type_sorted["TVT"], errors="coerce").to_numpy(np.float32)
    type_gr = fill_numeric(type_sorted["GR"])
    type_gr = rolling_mean(type_gr, int(audit_config.get("typewell_smooth_window", 5)))
    full_gr = fill_numeric(horizontal["GR"])
    filters = build_filters(full_gr, list(audit_config.get("filters", [{"name": "raw", "kind": "raw"}])))
    return WellBundle(
        well=well,
        horizontal=horizontal,
        typewell=typewell,
        md=md,
        true_tvt=true_tvt,
        tvt_input=tvt_input,
        known_end=known_end,
        full_gr=full_gr,
        type_tvt=type_tvt,
        type_gr=type_gr,
        filters=filters,
    )


def score_rows_for_filter(
    *,
    bundle: WellBundle,
    filter_name: str,
    region: str,
    row_idx: np.ndarray,
    region_known_end: int,
    prior: np.ndarray,
    shifts: np.ndarray,
    local_offsets: np.ndarray,
    ncc_weight: float,
    score_temperature: float,
) -> pd.DataFrame:
    if row_idx.size == 0:
        return pd.DataFrame()
    filtered = bundle.filters[filter_name]
    eval_gr = gather_horizontal(filtered.values, row_idx, local_offsets)
    local_rows = np.clip(row_idx[:, None] + local_offsets[None, :], 0, len(prior) - 1)
    local_prior = prior[local_rows]
    candidate_tvt = local_prior[:, None, :] + shifts[None, :, None]
    candidate_gr = interpolate_typewell(bundle.type_tvt, bundle.type_gr, candidate_tvt)
    mae = np.mean(np.abs(candidate_gr - eval_gr[:, None, :]), axis=2)
    ncc = np.mean(standardize_rows(candidate_gr) * standardize_rows(eval_gr)[:, None, :], axis=2)
    cost = mae - float(ncc_weight) * ncc
    best_pos = np.argmin(cost, axis=1)
    best_cost = cost[np.arange(len(row_idx)), best_pos]
    second_cost = np.partition(cost, 1, axis=1)[:, 1] if cost.shape[1] > 1 else best_cost
    logits = -(cost - cost.min(axis=1, keepdims=True)) / max(float(score_temperature), 1e-6)
    weights = np.exp(np.clip(logits, -80.0, 80.0))
    weights /= weights.sum(axis=1, keepdims=True) + 1e-12
    entropy = -np.sum(weights * np.log(weights + 1e-12), axis=1) / np.log(cost.shape[1])
    best_shift = shifts[best_pos]
    prior_center = prior[row_idx]
    pred_tvt = prior_center + best_shift
    true_shift = bundle.true_tvt[row_idx] - prior_center
    error = pred_tvt - bundle.true_tvt[row_idx]
    peak_periods = [
        peak.get("period_ft")
        for peak in filtered.metadata.get("peaks", [])
        if peak.get("period_ft") is not None
    ]
    return pd.DataFrame(
        {
            "id": [f"{bundle.well}_{int(idx)}" for idx in row_idx],
            "well": bundle.well,
            "eval_region": region,
            "filter": filter_name,
            "row_idx": row_idx.astype(np.int32),
            "md": bundle.md[row_idx].astype(np.float32),
            "known_prefix_rows": int(region_known_end),
            "distance_from_known_prefix": (row_idx - int(region_known_end)).astype(np.float32),
            "true_tvt": bundle.true_tvt[row_idx].astype(np.float32),
            "prior_center_tvt": prior_center.astype(np.float32),
            "true_shift_ft": true_shift.astype(np.float32),
            "best_shift_ft": best_shift.astype(np.float32),
            "matched_center_tvt": pred_tvt.astype(np.float32),
            "error": error.astype(np.float32),
            "abs_error": np.abs(error).astype(np.float32),
            "best_cost": best_cost.astype(np.float32),
            "top1_top2_cost_gap": (second_cost - best_cost).astype(np.float32),
            "entropy": entropy.astype(np.float32),
            "best_ncc": ncc[np.arange(len(row_idx)), best_pos].astype(np.float32),
            "best_gr_mae": mae[np.arange(len(row_idx)), best_pos].astype(np.float32),
            "fft_primary_period_ft": float(peak_periods[0]) if peak_periods else np.nan,
        }
    )


def score_matching_pairs(config: dict[str, Any], train_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_config = get_nested(config, "audit", {})
    wells = list_wells(train_dir, audit_config)
    max_eval = int(audit_config.get("max_eval_rows_per_region_per_well", 32))
    min_prefix = int(audit_config.get("min_prefix_rows", 80))
    prefix_tail = int(audit_config.get("prefix_backtest_tail_rows", 256))
    slope_window = int(audit_config.get("prefix_slope_window_rows", 80))
    slope_clip_values = audit_config.get("slope_clip", [-3.0, 3.0])
    slope_clip = (float(slope_clip_values[0]), float(slope_clip_values[1]))
    shifts = np.arange(
        float(audit_config.get("shift_min_ft", -220.0)),
        float(audit_config.get("shift_max_ft", 220.0)) + 0.5 * float(audit_config.get("shift_step_ft", 5.0)),
        float(audit_config.get("shift_step_ft", 5.0)),
        dtype=np.float32,
    )
    local_offsets = np.asarray(audit_config.get("local_offsets_rows", [-24, -12, 0, 12, 24]), dtype=np.int32)

    row_frames: list[pd.DataFrame] = []
    input_rows: list[dict[str, Any]] = []
    for index, well in enumerate(wells, start=1):
        print(f"[{index}/{len(wells)}] scoring {well}")
        try:
            bundle = load_well_bundle(well, train_dir, audit_config)
        except Exception as exc:
            input_rows.append({"well": well, "skipped": True, "reason": type(exc).__name__, "message": str(exc)})
            continue
        if bundle.known_end < min_prefix:
            input_rows.append(
                {
                    "well": well,
                    "skipped": True,
                    "reason": "short_prefix",
                    "known_prefix_rows": int(bundle.known_end),
                    "horizontal_rows": int(len(bundle.horizontal)),
                    "typewell_rows": int(len(bundle.typewell)),
                }
            )
            continue
        regions: list[tuple[str, int, np.ndarray]] = []
        hidden_rows = deterministic_eval_indices(bundle.known_end, len(bundle.horizontal), max_eval)
        if hidden_rows.size:
            regions.append(("hidden_tail", bundle.known_end, hidden_rows))
        backtest_start = max(min_prefix, bundle.known_end - prefix_tail)
        backtest_rows = deterministic_eval_indices(backtest_start, bundle.known_end, max_eval)
        if backtest_rows.size:
            regions.append(("prefix_backtest", backtest_start, backtest_rows))
        for region, region_known_end, row_idx in regions:
            prior, prior_meta = prefix_slope_prior(
                md=bundle.md,
                tvt_input=bundle.tvt_input,
                known_end=region_known_end,
                slope_window_rows=slope_window,
                slope_clip=slope_clip,
            )
            for filter_name, filtered in bundle.filters.items():
                frame = score_rows_for_filter(
                    bundle=bundle,
                    filter_name=filter_name,
                    region=region,
                    row_idx=row_idx,
                    region_known_end=region_known_end,
                    prior=prior,
                    shifts=shifts,
                    local_offsets=local_offsets,
                    ncc_weight=float(audit_config.get("ncc_weight", 8.0)),
                    score_temperature=float(audit_config.get("score_temperature", 6.0)),
                )
                if not frame.empty:
                    row_frames.append(frame)
                input_rows.append(
                    {
                        "well": well,
                        "filter": filter_name,
                        "eval_region": region,
                        "skipped": False,
                        "horizontal_rows": int(len(bundle.horizontal)),
                        "typewell_rows": int(len(bundle.typewell)),
                        "known_prefix_rows": int(region_known_end),
                        "eval_rows": int(len(row_idx)),
                        "gr_missing_rate": float(pd.isna(bundle.horizontal["GR"]).mean()),
                        "typewell_gr_missing_rate": float(pd.isna(bundle.typewell["GR"]).mean()),
                        "horizontal_sha256": sha256_file(train_dir / f"{well}__horizontal_well.csv"),
                        "typewell_sha256": sha256_file(train_dir / f"{well}__typewell.csv"),
                        "filter_metadata": json.dumps(to_jsonable(filtered.metadata), sort_keys=True),
                        **{f"prior_{key}": value for key, value in prior_meta.items()},
                    }
                )
    if not row_frames:
        raise RuntimeError("No GR matching pairs were scored")
    return pd.concat(row_frames, ignore_index=True), pd.DataFrame(input_rows)


# %% [markdown]
# ## 4. Pair selection and plotting helpers

# %%
def select_examples(scored_pairs: pd.DataFrame, audit_config: dict[str, Any]) -> pd.DataFrame:
    max_per_group = int(audit_config.get("max_examples_per_filter_region", 4))
    max_total = int(audit_config.get("max_total_figures", 48))
    selected: list[pd.Series] = []
    for (_, _,), group in scored_pairs.groupby(["filter", "eval_region"], sort=False):
        picks: list[pd.Series] = []
        candidates = [
            ("wrong_depth_ge15", group[group["abs_error"] >= 15.0].sort_values(["abs_error", "top1_top2_cost_gap", "id"], ascending=[False, True, True])),
            ("wrong_depth_ge10", group[(group["abs_error"] >= 10.0) & (group["abs_error"] < 15.0)].sort_values(["abs_error", "top1_top2_cost_gap", "id"], ascending=[False, True, True])),
            ("wrong_depth_ge6", group[(group["abs_error"] >= 6.0) & (group["abs_error"] < 10.0)].sort_values(["abs_error", "top1_top2_cost_gap", "id"], ascending=[False, True, True])),
            (
                "ambiguous_wrong_depth",
                group[group.get("ambiguous_wrong_depth_flag", False)]
                .sort_values(["top1_top2_cost_gap", "entropy", "abs_error", "id"], ascending=[True, False, False, True]),
            ),
            ("best_abs_error", group.sort_values(["abs_error", "entropy", "id"], ascending=[True, True, True])),
            ("worst_abs_error", group.sort_values(["abs_error", "id"], ascending=[False, True])),
            ("high_gap_good", group.sort_values(["top1_top2_cost_gap", "abs_error", "id"], ascending=[False, True, True])),
            ("ambiguous_low_gap", group.sort_values(["top1_top2_cost_gap", "entropy", "id"], ascending=[True, False, True])),
            (
                "large_prior_correction",
                group.assign(prior_abs_error=lambda x: np.abs(x["true_shift_ft"]))
                .assign(correction_gain=lambda x: x["prior_abs_error"] - x["abs_error"])
                .sort_values(["correction_gain", "id"], ascending=[False, True]),
            ),
        ]
        seen: set[str] = set()
        for reason, ordered in candidates:
            for _, row in ordered.iterrows():
                key = str(row["id"])
                if key in seen:
                    continue
                item = row.copy()
                item["selection_reason"] = reason
                picks.append(item)
                seen.add(key)
                break
            if len(picks) >= max_per_group:
                break
        selected.extend(picks[:max_per_group])
    if not selected:
        raise RuntimeError("No examples selected for visualization")
    out = pd.DataFrame(selected).reset_index(drop=True)
    out = out.head(max_total).copy()
    out.insert(0, "figure_index", np.arange(1, len(out) + 1, dtype=np.int32))
    return out


def row_matching_detail(
    *,
    bundle: WellBundle,
    row: pd.Series,
    audit_config: dict[str, Any],
) -> dict[str, Any]:
    filter_name = str(row["filter"])
    region_known_end = int(row["known_prefix_rows"])
    slope_window = int(audit_config.get("prefix_slope_window_rows", 80))
    slope_clip_values = audit_config.get("slope_clip", [-3.0, 3.0])
    prior, prior_meta = prefix_slope_prior(
        md=bundle.md,
        tvt_input=bundle.tvt_input,
        known_end=region_known_end,
        slope_window_rows=slope_window,
        slope_clip=(float(slope_clip_values[0]), float(slope_clip_values[1])),
    )
    shifts = np.arange(
        float(audit_config.get("shift_min_ft", -220.0)),
        float(audit_config.get("shift_max_ft", 220.0)) + 0.5 * float(audit_config.get("shift_step_ft", 5.0)),
        float(audit_config.get("shift_step_ft", 5.0)),
        dtype=np.float32,
    )
    local_offsets = np.asarray(audit_config.get("local_offsets_rows", [-24, -12, 0, 12, 24]), dtype=np.int32)
    idx = int(row["row_idx"])
    filtered = bundle.filters[filter_name].values
    local_rows = np.clip(idx + local_offsets, 0, len(filtered) - 1)
    horizontal_gr = filtered[local_rows]
    horizontal_raw = bundle.full_gr[local_rows]
    local_prior = prior[local_rows]
    candidate_tvt = local_prior[None, :] + shifts[:, None]
    candidate_gr = interpolate_typewell(bundle.type_tvt, bundle.type_gr, candidate_tvt)
    mae = np.mean(np.abs(candidate_gr - horizontal_gr[None, :]), axis=1)
    ncc = np.mean(standardize_rows(candidate_gr) * standardize_rows(horizontal_gr[None, :]), axis=1)
    cost = mae - float(audit_config.get("ncc_weight", 8.0)) * ncc
    best_pos = int(np.argmin(cost))
    matched_gr = candidate_gr[best_pos]
    true_gr = interpolate_typewell(bundle.type_tvt, bundle.type_gr, bundle.true_tvt[local_rows][None, :])[0]
    return {
        "prior": prior,
        "prior_meta": prior_meta,
        "shifts": shifts,
        "local_offsets": local_offsets,
        "local_rows": local_rows,
        "horizontal_gr": horizontal_gr,
        "horizontal_raw": horizontal_raw,
        "local_prior": local_prior,
        "cost": cost,
        "mae": mae,
        "ncc": ncc,
        "candidate_tvt": candidate_tvt,
        "candidate_gr": candidate_gr,
        "best_pos": best_pos,
        "matched_gr": matched_gr,
        "true_gr": true_gr,
        "filter_values": filtered,
    }


def standardized_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return ((values - float(values.mean())) / (float(values.std()) + 1e-6)).astype(np.float32)


def decimated_xy(x: np.ndarray, y: np.ndarray, max_points: int = 5000) -> tuple[np.ndarray, np.ndarray]:
    if len(x) <= int(max_points):
        return x, y
    step = int(np.ceil(len(x) / int(max_points)))
    return x[::step], y[::step]


def top_candidate_rows(
    *,
    row: pd.Series,
    detail: dict[str, Any],
    audit_config: dict[str, Any],
) -> list[dict[str, Any]]:
    return list(minima_diagnostics(row=row, detail=detail, audit_config=audit_config)["top_local_minima"])


def format_optional_float(value: Any, digits: int = 2) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return "NA"


def local_minimum_positions(cost: np.ndarray) -> np.ndarray:
    cost = np.asarray(cost, dtype=np.float32)
    if cost.size == 0:
        return np.zeros(0, dtype=np.int32)
    if cost.size == 1:
        return np.asarray([0], dtype=np.int32)
    positions: list[int] = []
    if cost[0] <= cost[1]:
        positions.append(0)
    for idx in range(1, cost.size - 1):
        if cost[idx] <= cost[idx - 1] and cost[idx] <= cost[idx + 1]:
            positions.append(idx)
    if cost[-1] <= cost[-2]:
        positions.append(cost.size - 1)
    return np.asarray(positions, dtype=np.int32)


def candidate_row_from_position(
    *,
    pos: int,
    rank: int,
    tag: str,
    row: pd.Series,
    detail: dict[str, Any],
    best_cost: float,
) -> dict[str, Any]:
    shifts = np.asarray(detail["shifts"], dtype=np.float32)
    cost = np.asarray(detail["cost"], dtype=np.float32)
    mae = np.asarray(detail["mae"], dtype=np.float32)
    ncc = np.asarray(detail["ncc"], dtype=np.float32)
    center_idx = int(np.argmin(np.abs(np.asarray(detail["local_rows"]) - int(row["row_idx"]))))
    shift = float(shifts[int(pos)])
    matched_center = float(detail["local_prior"][center_idx] + shift)
    error = matched_center - float(row["true_tvt"])
    return {
        "rank": int(rank),
        "tag": str(tag),
        "pos": int(pos),
        "shift_ft": shift,
        "matched_center_tvt": matched_center,
        "cost": float(cost[int(pos)]),
        "delta_cost": float(cost[int(pos)] - best_cost),
        "mae": float(mae[int(pos)]),
        "ncc": float(ncc[int(pos)]),
        "error_ft": error,
        "abs_error_ft": abs(error),
    }


def minima_diagnostics(
    *,
    row: pd.Series,
    detail: dict[str, Any],
    audit_config: dict[str, Any],
) -> dict[str, Any]:
    max_candidates = int(audit_config.get("top_k_alternatives", 6))
    min_sep = float(audit_config.get("alternative_min_shift_separation_ft", 10.0))
    true_near_window = float(audit_config.get("true_near_min_window_ft", 5.0))
    shifts = np.asarray(detail["shifts"], dtype=np.float32)
    cost = np.asarray(detail["cost"], dtype=np.float32)
    best_pos = int(np.nanargmin(cost))
    best_cost = float(cost[best_pos])
    minima = local_minimum_positions(cost)
    if best_pos not in set(int(p) for p in minima):
        minima = np.asarray(sorted([*minima.tolist(), best_pos]), dtype=np.int32)

    selected_positions: list[int] = []
    top_local_rows: list[dict[str, Any]] = []
    for pos in minima[np.argsort(cost[minima])]:
        shift = float(shifts[int(pos)])
        if any(abs(shift - float(shifts[prev])) < min_sep for prev in selected_positions):
            continue
        selected_positions.append(int(pos))
        top_local_rows.append(
            candidate_row_from_position(
                pos=int(pos),
                rank=len(top_local_rows) + 1,
                tag=f"local_min_{len(top_local_rows) + 1}",
                row=row,
                detail=detail,
                best_cost=best_cost,
            )
        )
        if len(top_local_rows) >= max_candidates:
            break

    true_shift = float(row["true_shift_ft"])
    true_near_minima = minima[np.abs(shifts[minima] - true_shift) <= true_near_window]
    if len(true_near_minima):
        true_pos = int(true_near_minima[np.argmin(cost[true_near_minima])])
        true_tag = f"true_near_min_within_{true_near_window:g}ft"
    else:
        true_pos = int(minima[np.argmin(np.abs(shifts[minima] - true_shift))])
        true_tag = "nearest_local_min_to_true"
    true_row = candidate_row_from_position(
        pos=true_pos,
        rank=0,
        tag=true_tag,
        row=row,
        detail=detail,
        best_cost=best_cost,
    )

    top2 = top_local_rows[1] if len(top_local_rows) > 1 else None
    top3 = top_local_rows[2] if len(top_local_rows) > 2 else None
    best_abs = float(row["abs_error"])
    return {
        "top_local_minima": top_local_rows,
        "true_near_minimum": true_row,
        "best_local_minimum": top_local_rows[0] if top_local_rows else None,
        "second_local_minimum": top2,
        "third_local_minimum": top3,
        "local_minima_count": int(len(minima)),
        "best_true_abs_error_ft": best_abs,
        "wrong_depth_ge_6ft": bool(best_abs >= 6.0),
        "wrong_depth_ge_10ft": bool(best_abs >= 10.0),
        "wrong_depth_ge_15ft": bool(best_abs >= 15.0),
        "top2_shift_gap_ft": None if top2 is None else abs(float(top2["shift_ft"]) - float(top_local_rows[0]["shift_ft"])),
        "top2_delta_cost": None if top2 is None else float(top2["delta_cost"]),
        "true_near_delta_cost": float(true_row["delta_cost"]),
        "true_near_shift_ft": float(true_row["shift_ft"]),
        "true_near_abs_error_ft": float(true_row["abs_error_ft"]),
        "true_near_tag": str(true_row["tag"]),
    }


def add_match_diagnostic_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    bins = [-np.inf, 6.0, 10.0, 15.0, np.inf]
    labels = ["lt6ft", "ge6_lt10ft", "ge10_lt15ft", "ge15ft"]
    out["wrong_depth_bucket"] = pd.cut(out["abs_error"], bins=bins, labels=labels, right=False).astype(str)
    out["wrong_depth_ge_6ft"] = out["abs_error"] >= 6.0
    out["wrong_depth_ge_10ft"] = out["abs_error"] >= 10.0
    out["wrong_depth_ge_15ft"] = out["abs_error"] >= 15.0
    out["low_gap_flag"] = out["top1_top2_cost_gap"] <= float(out["top1_top2_cost_gap"].quantile(0.25))
    out["high_entropy_flag"] = out["entropy"] >= float(out["entropy"].quantile(0.75))
    out["ambiguous_wrong_depth_flag"] = out["wrong_depth_ge_10ft"] & (out["low_gap_flag"] | out["high_entropy_flag"])
    return out


def build_oof_bucket_summary(scored_pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_specs = [
        ("wrong_depth_bucket", "wrong_depth_bucket"),
        ("wrong_depth_ge_6ft", "wrong_depth_ge_6ft"),
        ("wrong_depth_ge_10ft", "wrong_depth_ge_10ft"),
        ("wrong_depth_ge_15ft", "wrong_depth_ge_15ft"),
        ("ambiguous_wrong_depth_flag", "ambiguous_wrong_depth_flag"),
    ]
    for group_name, column in group_specs:
        if column not in scored_pairs:
            continue
        for value, group in scored_pairs.groupby(column, dropna=False, sort=False):
            oof = group["oof_abs_error"] if "oof_abs_error" in group else pd.Series(dtype=float)
            rows.append(
                {
                    "group": group_name,
                    "value": str(value),
                    "rows": int(len(group)),
                    "wells": int(group["well"].nunique()),
                    "gr_match_abs_mean": float(group["abs_error"].mean()),
                    "gr_match_abs_median": float(group["abs_error"].median()),
                    "oof_rows": int(oof.notna().sum()),
                    "oof_abs_mean": float(oof.mean()) if oof.notna().any() else np.nan,
                    "oof_abs_median": float(oof.median()) if oof.notna().any() else np.nan,
                    "oof_abs_p90": float(oof.quantile(0.90)) if oof.notna().any() else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_pair(
    *,
    bundle: WellBundle,
    row: pd.Series,
    detail: dict[str, Any],
    audit_config: dict[str, Any],
    figures_dir: Path,
) -> str:
    fig_idx = int(row["figure_index"])
    well = str(row["well"])
    region = str(row["eval_region"])
    filter_name = str(row["filter"])
    row_idx = int(row["row_idx"])
    filename = f"{fig_idx:03d}_{well}_{region}_{filter_name}_row{row_idx}.png".replace("/", "_")
    output_path = figures_dir / filename

    context_half_rows = int(audit_config.get("context_half_window_rows", 220))
    lo = max(0, row_idx - context_half_rows)
    hi = min(len(bundle.full_gr), row_idx + context_half_rows + 1)
    context_rows = np.arange(lo, hi)
    true_shift = float(row["true_shift_ft"])
    best_shift = float(row["best_shift_ft"])
    matched_center = float(row["matched_center_tvt"])
    prior_center = float(row["prior_center_tvt"])
    true_center = float(row["true_tvt"])
    half_tvt = float(audit_config.get("typewell_context_half_window_ft", 160.0))
    type_mask = (bundle.type_tvt >= matched_center - half_tvt) & (bundle.type_tvt <= matched_center + half_tvt)
    if not type_mask.any():
        type_mask = np.ones(len(bundle.type_tvt), dtype=bool)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(context_rows, bundle.full_gr[context_rows], color="0.70", linewidth=1.0, label="horizontal raw GR")
    ax.plot(context_rows, detail["filter_values"][context_rows], color="#1f77b4", linewidth=1.3, label=f"horizontal {filter_name}")
    ax.axvline(int(row["known_prefix_rows"]), color="#2ca02c", linestyle="--", linewidth=1.0, label="known prefix end")
    ax.axvline(row_idx, color="#d62728", linestyle="-", linewidth=1.2, label="eval row")
    ax.set_title("Horizontal GR context")
    ax.set_xlabel("horizontal row index")
    ax.set_ylabel("GR")
    ax.legend(loc="best", fontsize=8)

    ax = axes[0, 1]
    shifts = detail["shifts"]
    ax.plot(shifts, detail["cost"], color="#111111", linewidth=1.3, label="matching cost")
    ax.axvline(0.0, color="0.55", linestyle=":", linewidth=1.0, label="prior shift 0")
    ax.axvline(best_shift, color="#d62728", linestyle="-", linewidth=1.2, label=f"best {best_shift:.1f} ft")
    ax.axvline(true_shift, color="#2ca02c", linestyle="--", linewidth=1.2, label=f"true {true_shift:.1f} ft")
    ax.set_title("Shift scan")
    ax.set_xlabel("shift added to prior TVT (ft)")
    ax.set_ylabel("MAE - ncc_weight * NCC")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 0]
    offsets = detail["local_offsets"]
    ax.plot(offsets, standardized_vector(detail["horizontal_gr"]), marker="o", color="#1f77b4", label="horizontal filtered GR")
    ax.plot(offsets, standardized_vector(detail["matched_gr"]), marker="s", color="#d62728", label="matched typewell GR")
    ax.plot(offsets, standardized_vector(detail["true_gr"]), marker="^", color="#2ca02c", label="typewell at true TVT")
    ax.set_title("Local waveform overlay")
    ax.set_xlabel("row offset around eval row")
    ax.set_ylabel("z-scored GR within window")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 1]
    ax.plot(bundle.type_tvt[type_mask], bundle.type_gr[type_mask], color="#111111", linewidth=1.2, label="typewell smoothed GR")
    ax.axvline(prior_center, color="0.55", linestyle=":", linewidth=1.0, label=f"prior {prior_center:.1f}")
    ax.axvline(matched_center, color="#d62728", linestyle="-", linewidth=1.2, label=f"matched {matched_center:.1f}")
    ax.axvline(true_center, color="#2ca02c", linestyle="--", linewidth=1.2, label=f"true {true_center:.1f}")
    ax.set_title("Typewell TVT context")
    ax.set_xlabel("TVT")
    ax.set_ylabel("GR")
    ax.legend(loc="best", fontsize=8)

    fig.suptitle(
        (
            f"{fig_idx:03d} {well} {region} {filter_name} row={row_idx} "
            f"reason={row['selection_reason']} abs_error={float(row['abs_error']):.2f} "
            f"gap={float(row['top1_top2_cost_gap']):.2f} entropy={float(row['entropy']):.2f}"
        ),
        fontsize=12,
    )
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return str(output_path)


def write_html_index(selected_pairs: pd.DataFrame, index_path: Path, artifact_dir: Path) -> None:
    rows = [
        "<html><head><meta charset='utf-8'><title>GR matching pair visualization</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;} table{border-collapse:collapse;} td,th{border:1px solid #ccc;padding:4px 6px;font-size:12px;} img{max-width:1100px;width:100%;border:1px solid #ccc;margin:10px 0 28px;} .meta{margin-bottom:8px;}</style>",
        "</head><body>",
        "<h1>GR matching pair visualization</h1>",
        "<table><thead><tr><th>#</th><th>well</th><th>region</th><th>filter</th><th>row</th><th>reason</th><th>best shift</th><th>true shift</th><th>abs error</th><th>gap</th><th>entropy</th></tr></thead><tbody>",
    ]
    for _, row in selected_pairs.iterrows():
        rel = Path(str(row["figure_path"])).relative_to(artifact_dir)
        rows.append(
            "<tr>"
            f"<td>{int(row['figure_index'])}</td>"
            f"<td>{html.escape(str(row['well']))}</td>"
            f"<td>{html.escape(str(row['eval_region']))}</td>"
            f"<td>{html.escape(str(row['filter']))}</td>"
            f"<td>{int(row['row_idx'])}</td>"
            f"<td>{html.escape(str(row['selection_reason']))}</td>"
            f"<td>{float(row['best_shift_ft']):.2f}</td>"
            f"<td>{float(row['true_shift_ft']):.2f}</td>"
            f"<td>{float(row['abs_error']):.2f}</td>"
            f"<td>{float(row['top1_top2_cost_gap']):.2f}</td>"
            f"<td>{float(row['entropy']):.2f}</td>"
            "</tr>"
        )
    rows.append("</tbody></table>")
    for _, row in selected_pairs.iterrows():
        rel = Path(str(row["figure_path"])).relative_to(artifact_dir)
        rows.append(
            f"<h2>{int(row['figure_index']):03d} {html.escape(str(row['well']))} "
            f"{html.escape(str(row['eval_region']))} {html.escape(str(row['filter']))}</h2>"
        )
        rows.append(
            f"<div class='meta'>row={int(row['row_idx'])}, reason={html.escape(str(row['selection_reason']))}, "
            f"abs_error={float(row['abs_error']):.2f}, gap={float(row['top1_top2_cost_gap']):.2f}, "
            f"entropy={float(row['entropy']):.2f}</div>"
        )
        rows.append(f"<img src='{html.escape(str(rel))}' alt='figure {int(row['figure_index'])}'>")
    rows.append("</body></html>")
    index_path.write_text("\n".join(rows))


def find_oof_prediction_path(config: dict[str, Any], artifact_dir: Path) -> Path | None:
    oof_config = get_nested(config, "audit.oof", {})
    if not bool(oof_config.get("enabled", False)):
        return None
    filename = str(oof_config.get("filename", "")).strip()
    candidates: list[Path] = []
    local_path = oof_config.get("local_path")
    if local_path:
        candidates.append(Path(str(local_path)))
        candidates.append(find_project_root() / str(local_path))
    if filename:
        candidates.extend(
            [
                artifact_dir / filename,
                Path.cwd() / filename,
                Path.cwd() / "artifacts" / filename,
                find_project_root() / filename,
            ]
        )
        if KAGGLE_INPUT_ROOT.exists():
            candidates.extend(sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}")))
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve() if candidate.exists() else candidate
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def load_oof_errors_for_ids(oof_path: Path | None, ids: set[str], model_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if oof_path is None:
        return pd.DataFrame(columns=["id"]), {"available": False, "reason": "oof_prediction_file_not_found"}
    usecols = ["id", "well", "model", "target_tvt", "pred_tvt"]
    rows: list[pd.DataFrame] = []
    total_rows = 0
    matched_rows = 0
    for chunk in pd.read_csv(oof_path, usecols=usecols, chunksize=250_000, low_memory=False):
        total_rows += len(chunk)
        chunk = chunk[chunk["model"].astype(str).eq(str(model_name))]
        chunk = chunk[chunk["id"].astype(str).isin(ids)]
        if chunk.empty:
            continue
        matched_rows += len(chunk)
        rows.append(chunk.copy())
    if not rows:
        return pd.DataFrame(columns=["id"]), {
            "available": True,
            "path": str(oof_path),
            "model": str(model_name),
            "total_rows_scanned": int(total_rows),
            "matched_rows": 0,
            "reason": "no_selected_pair_ids_matched",
        }
    oof = pd.concat(rows, ignore_index=True)
    oof["id"] = oof["id"].astype(str)
    oof["target_tvt"] = pd.to_numeric(oof["target_tvt"], errors="coerce")
    oof["pred_tvt"] = pd.to_numeric(oof["pred_tvt"], errors="coerce")
    oof["oof_error"] = oof["pred_tvt"] - oof["target_tvt"]
    oof["oof_abs_error"] = np.abs(oof["oof_error"])
    oof = oof.sort_values(["oof_abs_error", "id"], ascending=[True, True]).drop_duplicates("id", keep="first")
    return oof[["id", "target_tvt", "pred_tvt", "oof_error", "oof_abs_error"]], {
        "available": True,
        "path": str(oof_path),
        "model": str(model_name),
        "total_rows_scanned": int(total_rows),
        "matched_rows": int(matched_rows),
        "unique_ids_matched": int(oof["id"].nunique()),
    }


def plot_simple_overlay(
    *,
    row: pd.Series,
    detail: dict[str, Any],
    simple_dir: Path,
) -> str:
    fig_idx = int(row["figure_index"])
    well = str(row["well"])
    region = str(row["eval_region"])
    filter_name = str(row["filter"])
    row_idx = int(row["row_idx"])
    filename = f"{fig_idx:03d}_{well}_{region}_{filter_name}_row{row_idx}_overlay.png".replace("/", "_")
    output_path = simple_dir / filename

    offsets = detail["local_offsets"]
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.2), constrained_layout=True)
    ax.plot(
        offsets,
        standardized_vector(detail["horizontal_gr"]),
        marker="o",
        linewidth=2.0,
        color="#1f77b4",
        label="query: horizontal GR",
    )
    ax.plot(
        offsets,
        standardized_vector(detail["matched_gr"]),
        marker="s",
        linewidth=2.0,
        color="#d62728",
        label="matched: typewell GR",
    )
    ax.plot(
        offsets,
        standardized_vector(detail["true_gr"]),
        marker="^",
        linewidth=1.4,
        color="#2ca02c",
        alpha=0.65,
        label="typewell at true TVT",
    )
    oof_text = ""
    if "oof_abs_error" in row and pd.notna(row["oof_abs_error"]):
        oof_text = f" | OOF abs={float(row['oof_abs_error']):.2f}"
    ax.set_title(
        (
            f"{fig_idx:03d} {well} {region} {filter_name} row={row_idx} "
            f"match abs={float(row['abs_error']):.2f}{oof_text}"
        ),
        fontsize=11,
    )
    ax.set_xlabel("row offset around query row")
    ax.set_ylabel("z-scored GR within local window")
    ax.axhline(0.0, color="0.80", linewidth=0.8)
    ax.legend(loc="best", fontsize=9)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return str(output_path)


def plot_global_local_match(
    *,
    bundle: WellBundle,
    row: pd.Series,
    detail: dict[str, Any],
    audit_config: dict[str, Any],
    global_local_dir: Path,
) -> tuple[str, str]:
    fig_idx = int(row["figure_index"])
    well = str(row["well"])
    region = str(row["eval_region"])
    filter_name = str(row["filter"])
    row_idx = int(row["row_idx"])
    filename = f"{fig_idx:03d}_{well}_{region}_{filter_name}_row{row_idx}_global_local.png".replace("/", "_")
    output_path = global_local_dir / filename
    diag = minima_diagnostics(row=row, detail=detail, audit_config=audit_config)
    top_rows = list(diag["top_local_minima"])
    true_min = dict(diag["true_near_minimum"])

    local_rows = np.asarray(detail["local_rows"], dtype=np.int32)
    local_lo = int(local_rows.min())
    local_hi = int(local_rows.max())
    h_rows = np.arange(len(bundle.full_gr), dtype=np.int32)
    h_x, h_raw = decimated_xy(h_rows, bundle.full_gr, max_points=6000)
    _, h_filtered = decimated_xy(h_rows, detail["filter_values"], max_points=6000)
    t_x, t_gr = decimated_xy(bundle.type_tvt, bundle.type_gr, max_points=7000)

    true_shift = float(row["true_shift_ft"])
    best_shift = float(row["best_shift_ft"])
    matched_center = float(row["matched_center_tvt"])
    prior_center = float(row["prior_center_tvt"])
    true_center = float(row["true_tvt"])
    oof_text = format_optional_float(row.get("oof_abs_error", np.nan), digits=2)

    fig, axes = plt.subplots(3, 2, figsize=(17, 14), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(h_x, h_raw, color="0.78", linewidth=0.7, label="horizontal raw GR")
    ax.plot(h_x, h_filtered, color="#1f77b4", linewidth=0.9, label=f"horizontal {filter_name}")
    ax.axvspan(local_lo, local_hi, color="#ff7f0e", alpha=0.16, label="local query window")
    ax.axvline(int(row["known_prefix_rows"]), color="#2ca02c", linestyle="--", linewidth=1.0, label="known prefix end")
    ax.axvline(row_idx, color="#d62728", linewidth=1.0, label="query row")
    ax.set_title("Full horizontal GR and selected local window")
    ax.set_xlabel("horizontal row index")
    ax.set_ylabel("GR")
    ax.legend(loc="best", fontsize=8)

    ax = axes[0, 1]
    ax.plot(t_x, t_gr, color="#111111", linewidth=0.8, label="typewell GR")
    for item in top_rows[1:]:
        ax.axvline(float(item["matched_center_tvt"]), color="0.55", linestyle=":", linewidth=0.8, alpha=0.8)
    ax.axvline(prior_center, color="#9467bd", linestyle=":", linewidth=1.1, label=f"prior {prior_center:.1f}")
    ax.axvline(matched_center, color="#d62728", linewidth=1.3, label=f"best match {matched_center:.1f}")
    ax.axvline(float(true_min["matched_center_tvt"]), color="#17becf", linestyle="-.", linewidth=1.1, label=f"true-near min {float(true_min['matched_center_tvt']):.1f}")
    ax.axvline(true_center, color="#2ca02c", linestyle="--", linewidth=1.3, label=f"true {true_center:.1f}")
    ax.set_title("Full typewell GR with best and alternative centers")
    ax.set_xlabel("TVT")
    ax.set_ylabel("GR")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 0]
    offsets = detail["local_offsets"]
    ax.plot(offsets, standardized_vector(detail["horizontal_gr"]), marker="o", linewidth=2.0, color="#1f77b4", label="query horizontal GR")
    ax.plot(offsets, standardized_vector(detail["matched_gr"]), marker="s", linewidth=2.0, color="#d62728", label="best matched typewell GR")
    ax.plot(offsets, standardized_vector(detail["true_gr"]), marker="^", linewidth=1.4, color="#2ca02c", alpha=0.65, label="typewell at true TVT")
    ax.axhline(0.0, color="0.82", linewidth=0.8)
    ax.set_title("Local waveform overlay")
    ax.set_xlabel("row offset around query row")
    ax.set_ylabel("z-scored GR")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 1]
    ax.plot(offsets, standardized_vector(detail["horizontal_gr"]), marker="o", linewidth=2.0, color="#1f77b4", label="query horizontal GR")
    for item in top_rows[1:]:
        alt = detail["candidate_gr"][int(item["pos"])]
        ax.plot(offsets, standardized_vector(alt), color="0.55", linewidth=1.0, alpha=0.55)
    if len(top_rows) > 1:
        ax.plot([], [], color="0.55", linewidth=1.0, alpha=0.55, label="alternative candidates")
    if int(true_min["pos"]) != int(top_rows[0]["pos"]):
        ax.plot(
            offsets,
            standardized_vector(detail["candidate_gr"][int(true_min["pos"])]),
            color="#17becf",
            linewidth=1.6,
            linestyle="-.",
            alpha=0.9,
            label="true-near local minimum",
        )
    ax.plot(offsets, standardized_vector(detail["matched_gr"]), marker="s", linewidth=2.0, color="#d62728", label="best candidate")
    ax.axhline(0.0, color="0.82", linewidth=0.8)
    ax.set_title("Does the query also match nearby decoys?")
    ax.set_xlabel("row offset around query row")
    ax.set_ylabel("z-scored GR")
    ax.legend(loc="best", fontsize=8)

    ax = axes[2, 0]
    shifts = detail["shifts"]
    ax.plot(shifts, detail["cost"], color="#111111", linewidth=1.2, label="matching cost")
    for item in top_rows[1:]:
        ax.axvline(float(item["shift_ft"]), color="0.55", linestyle=":", linewidth=0.8, alpha=0.8)
    ax.axvline(0.0, color="#9467bd", linestyle=":", linewidth=1.0, label="prior shift 0")
    ax.axvline(best_shift, color="#d62728", linewidth=1.3, label=f"best {best_shift:.1f} ft")
    ax.axvline(float(true_min["shift_ft"]), color="#17becf", linestyle="-.", linewidth=1.2, label=f"true-near min {float(true_min['shift_ft']):.1f} ft")
    ax.axvline(true_shift, color="#2ca02c", linestyle="--", linewidth=1.3, label=f"true {true_shift:.1f} ft")
    ax.set_title("Shift scan and near-best alternatives")
    ax.set_xlabel("shift added to prior TVT (ft)")
    ax.set_ylabel("MAE - ncc_weight * NCC")
    ax.legend(loc="best", fontsize=8)

    ax = axes[2, 1]
    ax.axis("off")
    lines = [
        "Top local minima",
        "rank  shift  TVT_center  cost  d_cost  match_abs",
    ]
    for item in top_rows:
        lines.append(
            f"{int(item['rank']):>4} "
            f"{float(item['shift_ft']):>6.1f} "
            f"{float(item['matched_center_tvt']):>10.1f} "
            f"{float(item['cost']):>7.2f} "
            f"{float(item['delta_cost']):>7.2f} "
            f"{float(item['abs_error_ft']):>9.2f}"
        )
    lines.extend(
        [
            "",
            "True-near minimum",
            f"shift={float(true_min['shift_ft']):.1f}, d_cost={float(true_min['delta_cost']):.2f}, "
            f"match_abs={float(true_min['abs_error_ft']):.2f}, tag={true_min['tag']}",
        ]
    )
    lines.extend(
        [
            "",
            f"GR match abs error: {float(row['abs_error']):.2f}",
            f"OOF abs error: {oof_text}",
            f"top1-top2 cost gap: {float(row['top1_top2_cost_gap']):.2f}",
            f"entropy: {float(row['entropy']):.2f}",
            f"wrong-depth flags: >=6 {bool(diag['wrong_depth_ge_6ft'])}, >=10 {bool(diag['wrong_depth_ge_10ft'])}, >=15 {bool(diag['wrong_depth_ge_15ft'])}",
        ]
    )
    ax.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=10)

    fig.suptitle(
        (
            f"{fig_idx:03d} {well} {region} {filter_name} row={row_idx} "
            f"reason={row['selection_reason']} | GR match abs={float(row['abs_error']):.2f} | OOF abs={oof_text}"
        ),
        fontsize=13,
    )
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return str(output_path), json.dumps(to_jsonable(diag), sort_keys=True)


def write_global_local_html(selected_pairs: pd.DataFrame, index_path: Path, artifact_dir: Path) -> None:
    rows = [
        "<html><head><meta charset='utf-8'><title>GR matching global/local view</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;} .card{margin:0 0 34px 0;} .meta{font-size:13px;line-height:1.45;margin:6px 0 8px;} img{max-width:1500px;width:100%;border:1px solid #ccc;} table{border-collapse:collapse;margin-bottom:20px;} td,th{border:1px solid #ccc;padding:4px 6px;font-size:12px;}</style>",
        "</head><body>",
        "<h1>GR matching global + local diagnostics</h1>",
        "<p>Blue is the query horizontal GR. Red is the best matched typewell local minimum. Gray vertical lines / traces are other near-best local minima. Cyan is the local minimum nearest the train-only true shift. Green is the train-only true TVT reference.</p>",
        "<table><thead><tr><th>#</th><th>well</th><th>region</th><th>filter</th><th>row</th><th>reason</th><th>bucket</th><th>GR abs</th><th>true-near dCost</th><th>OOF abs</th><th>gap</th><th>entropy</th></tr></thead><tbody>",
    ]
    for _, row in selected_pairs.iterrows():
        rows.append(
            "<tr>"
            f"<td>{int(row['figure_index'])}</td>"
            f"<td>{html.escape(str(row['well']))}</td>"
            f"<td>{html.escape(str(row['eval_region']))}</td>"
            f"<td>{html.escape(str(row['filter']))}</td>"
            f"<td>{int(row['row_idx'])}</td>"
            f"<td>{html.escape(str(row['selection_reason']))}</td>"
            f"<td>{html.escape(str(row.get('wrong_depth_bucket', 'NA')))}</td>"
            f"<td>{float(row['abs_error']):.2f}</td>"
            f"<td>{html.escape(format_optional_float(row.get('true_near_delta_cost', np.nan), digits=2))}</td>"
            f"<td>{html.escape(format_optional_float(row.get('oof_abs_error', np.nan), digits=2))}</td>"
            f"<td>{float(row['top1_top2_cost_gap']):.2f}</td>"
            f"<td>{float(row['entropy']):.2f}</td>"
            "</tr>"
        )
    rows.append("</tbody></table>")
    for _, row in selected_pairs.iterrows():
        rel = Path(str(row["global_local_path"])).relative_to(artifact_dir)
        rows.append("<div class='card'>")
        rows.append(
            f"<h2>{int(row['figure_index']):03d} {html.escape(str(row['well']))} "
            f"{html.escape(str(row['eval_region']))} {html.escape(str(row['filter']))}</h2>"
        )
        rows.append(
            f"<div class='meta'>row={int(row['row_idx'])}, reason={html.escape(str(row['selection_reason']))}, "
            f"bucket={html.escape(str(row.get('wrong_depth_bucket', 'NA')))}, "
            f"GR match abs={float(row['abs_error']):.2f}, "
            f"true-near dCost={html.escape(format_optional_float(row.get('true_near_delta_cost', np.nan), digits=2))}, "
            f"OOF abs={html.escape(format_optional_float(row.get('oof_abs_error', np.nan), digits=2))}, "
            f"best shift={float(row['best_shift_ft']):.1f}, true shift={float(row['true_shift_ft']):.1f}</div>"
        )
        rows.append(f"<img src='{html.escape(str(rel))}' alt='global local figure {int(row['figure_index'])}'>")
        rows.append("</div>")
    rows.append("</body></html>")
    index_path.write_text("\n".join(rows))


def write_wrong_depth_html(
    selected_pairs: pd.DataFrame,
    bucket_summary: pd.DataFrame,
    index_path: Path,
    artifact_dir: Path,
) -> None:
    ordered = selected_pairs.sort_values(
        ["wrong_depth_ge_15ft", "wrong_depth_ge_10ft", "wrong_depth_ge_6ft", "abs_error", "oof_abs_error"],
        ascending=[False, False, False, False, False],
    )
    rows = [
        "<html><head><meta charset='utf-8'><title>GR matching wrong-depth diagnostics</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(560px,1fr));gap:18px;} .card{border:1px solid #ccc;padding:10px;} .meta{font-size:13px;line-height:1.45;margin-bottom:8px;} img{width:100%;border:1px solid #ddd;} table{border-collapse:collapse;margin:10px 0 20px;} td,th{border:1px solid #ccc;padding:4px 6px;font-size:12px;}</style>",
        "</head><body>",
        "<h1>GR matching wrong-depth / local-minimum diagnostics</h1>",
        "<p>This view is for the question: is the best GR fit the wrong depth? Red is the best local minimum, gray are top alternatives, cyan is the true-near local minimum, and green is the train-only true reference.</p>",
        "<h2>OOF comparison by GR wrong-depth bucket</h2>",
        "<table><thead><tr><th>group</th><th>value</th><th>rows</th><th>wells</th><th>GR abs mean</th><th>GR abs median</th><th>OOF rows</th><th>OOF abs mean</th><th>OOF abs median</th><th>OOF abs p90</th></tr></thead><tbody>",
    ]
    for _, row in bucket_summary.iterrows():
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row['group']))}</td>"
            f"<td>{html.escape(str(row['value']))}</td>"
            f"<td>{int(row['rows'])}</td>"
            f"<td>{int(row['wells'])}</td>"
            f"<td>{float(row['gr_match_abs_mean']):.2f}</td>"
            f"<td>{float(row['gr_match_abs_median']):.2f}</td>"
            f"<td>{int(row['oof_rows'])}</td>"
            f"<td>{html.escape(format_optional_float(row.get('oof_abs_mean', np.nan), digits=2))}</td>"
            f"<td>{html.escape(format_optional_float(row.get('oof_abs_median', np.nan), digits=2))}</td>"
            f"<td>{html.escape(format_optional_float(row.get('oof_abs_p90', np.nan), digits=2))}</td>"
            "</tr>"
        )
    rows.append("</tbody></table>")
    rows.append("<div class='grid'>")
    for _, row in ordered.iterrows():
        rel = Path(str(row["global_local_path"])).relative_to(artifact_dir)
        rows.append("<div class='card'>")
        rows.append(
            f"<div class='meta'><b>{int(row['figure_index']):03d}</b> {html.escape(str(row['well']))} "
            f"{html.escape(str(row['eval_region']))} {html.escape(str(row['filter']))} row={int(row['row_idx'])}<br>"
            f"reason={html.escape(str(row['selection_reason']))}, bucket={html.escape(str(row.get('wrong_depth_bucket', 'NA')))}<br>"
            f"GR abs={float(row['abs_error']):.2f}, OOF abs={html.escape(format_optional_float(row.get('oof_abs_error', np.nan), digits=2))}, "
            f"top2 dCost={html.escape(format_optional_float(row.get('top2_delta_cost', np.nan), digits=2))}, "
            f"true-near dCost={html.escape(format_optional_float(row.get('true_near_delta_cost', np.nan), digits=2))}</div>"
        )
        rows.append(f"<img src='{html.escape(str(rel))}' alt='wrong depth figure {int(row['figure_index'])}'>")
        rows.append("</div>")
    rows.append("</div></body></html>")
    index_path.write_text("\n".join(rows))


def write_good_bad_html(selected_pairs: pd.DataFrame, index_path: Path, artifact_dir: Path, oof_label: str) -> None:
    metric_col = "oof_abs_error" if "oof_abs_error" in selected_pairs and selected_pairs["oof_abs_error"].notna().any() else "abs_error"
    metric_label = oof_label if metric_col == "oof_abs_error" else "gr_match_abs_error"
    max_rows = min(12, max(1, len(selected_pairs) // 2))
    good = selected_pairs.sort_values([metric_col, "abs_error", "id"], ascending=[True, True, True]).head(max_rows)
    bad = selected_pairs.sort_values([metric_col, "abs_error", "id"], ascending=[False, False, True]).head(max_rows)

    def card_rows(frame: pd.DataFrame, title: str) -> list[str]:
        rows = [f"<h2>{html.escape(title)}</h2>"]
        rows.append("<div class='grid'>")
        for _, row in frame.iterrows():
            rel = Path(str(row["simple_overlay_path"])).relative_to(artifact_dir)
            oof_value = "NA" if pd.isna(row.get("oof_abs_error", np.nan)) else f"{float(row['oof_abs_error']):.2f}"
            rows.append("<div class='card'>")
            rows.append(
                f"<div class='meta'><b>{int(row['figure_index']):03d}</b> {html.escape(str(row['well']))} "
                f"{html.escape(str(row['eval_region']))} {html.escape(str(row['filter']))} row={int(row['row_idx'])}<br>"
                f"reason={html.escape(str(row['selection_reason']))}<br>"
                f"GR match abs={float(row['abs_error']):.2f}, OOF abs={oof_value}, "
                f"best shift={float(row['best_shift_ft']):.1f}, true shift={float(row['true_shift_ft']):.1f}</div>"
            )
            rows.append(f"<img src='{html.escape(str(rel))}' alt='simple overlay {int(row['figure_index'])}'>")
            rows.append("</div>")
        rows.append("</div>")
        return rows

    rows = [
        "<html><head><meta charset='utf-8'><title>GR matching good/bad overlays</title>",
        "<style>body{font-family:Arial,sans-serif;margin:24px;} .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px;} .card{border:1px solid #ccc;padding:10px;} .meta{font-size:13px;line-height:1.45;margin-bottom:8px;} img{width:100%;border:1px solid #ddd;} table{border-collapse:collapse;} td,th{border:1px solid #ccc;padding:4px 6px;font-size:12px;}</style>",
        "</head><body>",
        "<h1>GR matching query vs matched overlays</h1>",
        f"<p>Good/bad ordering metric: <b>{html.escape(metric_label)}</b>. Blue is the query horizontal GR window. Red is the matched typewell GR window. Green is the typewell GR at true TVT, shown only for train-side interpretation.</p>",
    ]
    rows.extend(card_rows(good, "Good matches / low error"))
    rows.extend(card_rows(bad, "Bad matches / high error"))
    rows.append("</body></html>")
    index_path.write_text("\n".join(rows))


# %% [markdown]
# ## 5. Setup and input checks

# %%
start_time = time.time()
config = load_config()
audit_config = get_nested(config, "audit", {})
artifact_dir, figures_dir = resolve_output_dirs()
train_dir = resolve_train_dir(config)

print("experiment:", get_nested(config, "experiment.name", EXPERIMENT_NAME))
print("route:", get_nested(config, "experiment.route"))
print("train_dir:", train_dir)
print("artifact_dir:", artifact_dir)
print("filters:", [spec["name"] for spec in audit_config.get("filters", [])])
print("max_wells:", audit_config.get("max_wells"))
print("max_eval_rows_per_region_per_well:", audit_config.get("max_eval_rows_per_region_per_well"))
print("max_total_figures:", audit_config.get("max_total_figures"))

sample_horizontal = sorted(train_dir.glob("*__horizontal_well.csv"))[:5]
print("sample horizontal files:", [path.name for path in sample_horizontal])
if not sample_horizontal:
    raise FileNotFoundError(f"No horizontal well files found in {train_dir}")

# %% [markdown]
# ## 6. Score GR matching pairs

# %%
scored_pairs, input_summary = score_matching_pairs(config, train_dir)
scored_pairs = add_match_diagnostic_columns(scored_pairs)
oof_config = get_nested(config, "audit.oof", {})
oof_path = find_oof_prediction_path(config, artifact_dir)
oof_errors, oof_summary = load_oof_errors_for_ids(
    oof_path,
    set(scored_pairs["id"].astype(str)),
    str(oof_config.get("model", "lgb1")),
)
if not oof_errors.empty:
    scored_pairs = scored_pairs.merge(oof_errors, on="id", how="left", validate="many_to_one")
else:
    scored_pairs["target_tvt"] = np.nan
    scored_pairs["pred_tvt"] = np.nan
    scored_pairs["oof_error"] = np.nan
    scored_pairs["oof_abs_error"] = np.nan
print("OOF join summary:", json.dumps(to_jsonable(oof_summary), indent=2, sort_keys=True))
oof_bucket_summary = build_oof_bucket_summary(scored_pairs)
display(scored_pairs.head())
display(
    scored_pairs.groupby(["filter", "eval_region"], as_index=False)
    .agg(
        rows=("id", "size"),
        wells=("well", "nunique"),
        rmse_tvt=("error", lambda x: float(np.sqrt(np.mean(np.square(x))))),
        mae_tvt=("abs_error", "mean"),
        gap_mean=("top1_top2_cost_gap", "mean"),
        entropy_mean=("entropy", "mean"),
        oof_rows=("oof_abs_error", lambda x: int(pd.notna(x).sum())),
        oof_abs_mean=("oof_abs_error", "mean"),
    )
    .sort_values(["eval_region", "rmse_tvt", "filter"])
)
display(oof_bucket_summary)

# %% [markdown]
# ## 7. Generate visualizations

# %%
selected_pairs = select_examples(scored_pairs, audit_config)
figure_paths: list[str] = []
bundle_cache: dict[str, WellBundle] = {}
for _, row in selected_pairs.iterrows():
    well = str(row["well"])
    if well not in bundle_cache:
        bundle_cache[well] = load_well_bundle(well, train_dir, audit_config)
    detail = row_matching_detail(bundle=bundle_cache[well], row=row, audit_config=audit_config)
    figure_paths.append(
        plot_pair(
            bundle=bundle_cache[well],
            row=row,
            detail=detail,
            audit_config=audit_config,
            figures_dir=figures_dir,
        )
    )
selected_pairs["figure_path"] = figure_paths

simple_overlay_dir = artifact_dir / "simple_overlay"
simple_overlay_dir.mkdir(parents=True, exist_ok=True)
simple_paths: list[str] = []
for _, row in selected_pairs.iterrows():
    well = str(row["well"])
    if well not in bundle_cache:
        bundle_cache[well] = load_well_bundle(well, train_dir, audit_config)
    detail = row_matching_detail(bundle=bundle_cache[well], row=row, audit_config=audit_config)
    simple_paths.append(plot_simple_overlay(row=row, detail=detail, simple_dir=simple_overlay_dir))
selected_pairs["simple_overlay_path"] = simple_paths

global_local_dir = artifact_dir / "global_local"
global_local_dir.mkdir(parents=True, exist_ok=True)
global_local_paths: list[str] = []
global_local_top_candidates: list[str] = []
for _, row in selected_pairs.iterrows():
    well = str(row["well"])
    if well not in bundle_cache:
        bundle_cache[well] = load_well_bundle(well, train_dir, audit_config)
    detail = row_matching_detail(bundle=bundle_cache[well], row=row, audit_config=audit_config)
    path, top_candidates_json = plot_global_local_match(
        bundle=bundle_cache[well],
        row=row,
        detail=detail,
        audit_config=audit_config,
        global_local_dir=global_local_dir,
    )
    global_local_paths.append(path)
    global_local_top_candidates.append(top_candidates_json)
selected_pairs["global_local_path"] = global_local_paths
selected_pairs["global_local_top_candidates"] = global_local_top_candidates
global_local_diag = pd.DataFrame([json.loads(value) for value in global_local_top_candidates])
for column in [
    "local_minima_count",
    "best_true_abs_error_ft",
    "wrong_depth_ge_6ft",
    "wrong_depth_ge_10ft",
    "wrong_depth_ge_15ft",
    "top2_shift_gap_ft",
    "top2_delta_cost",
    "true_near_delta_cost",
    "true_near_shift_ft",
    "true_near_abs_error_ft",
    "true_near_tag",
]:
    selected_pairs[column] = global_local_diag[column].to_numpy()

display(
    selected_pairs[
        [
            "figure_index",
            "well",
            "eval_region",
            "filter",
            "row_idx",
            "selection_reason",
            "abs_error",
            "wrong_depth_bucket",
            "true_near_delta_cost",
            "oof_abs_error",
            "top1_top2_cost_gap",
            "global_local_path",
            "simple_overlay_path",
            "figure_path",
        ]
    ]
)

# %% [markdown]
# ## 8. Summary and generated artifacts

# %%
scored_path = artifact_dir / f"{OUTPUT_PREFIX}_scored_pairs.csv.gz"
selected_path = artifact_dir / f"{OUTPUT_PREFIX}_selected_pairs.csv"
input_summary_path = artifact_dir / f"{OUTPUT_PREFIX}_input_summary.csv"
oof_bucket_summary_path = artifact_dir / f"{OUTPUT_PREFIX}_oof_bucket_summary.csv"
index_path = artifact_dir / f"{OUTPUT_PREFIX}_index.html"
good_bad_index_path = artifact_dir / f"{OUTPUT_PREFIX}_good_bad_oof_index.html"
global_local_index_path = artifact_dir / f"{OUTPUT_PREFIX}_global_local_index.html"
wrong_depth_index_path = artifact_dir / f"{OUTPUT_PREFIX}_wrong_depth_index.html"
summary_path = artifact_dir / f"{OUTPUT_PREFIX}_summary.json"

scored_pairs.to_csv(scored_path, index=False)
selected_pairs.to_csv(selected_path, index=False)
input_summary.to_csv(input_summary_path, index=False)
oof_bucket_summary.to_csv(oof_bucket_summary_path, index=False)
write_html_index(selected_pairs, index_path, artifact_dir)
write_good_bad_html(
    selected_pairs,
    good_bad_index_path,
    artifact_dir,
    str(oof_config.get("label", "oof_abs_error")),
)
write_global_local_html(selected_pairs, global_local_index_path, artifact_dir)
write_wrong_depth_html(selected_pairs, oof_bucket_summary, wrong_depth_index_path, artifact_dir)

summary = {
    "experiment": EXPERIMENT_NAME,
    "status": "completed_visualization_diagnostic",
    "route": get_nested(config, "experiment.route", "pf_beam"),
    "train_dir": str(train_dir),
    "rows_scored": int(len(scored_pairs)),
    "wells_scored": int(scored_pairs["well"].nunique()),
    "filters": sorted(scored_pairs["filter"].unique().tolist()),
    "eval_regions": sorted(scored_pairs["eval_region"].unique().tolist()),
    "figures": int(len(selected_pairs)),
    "artifacts": {
        "scored_pairs": str(scored_path),
        "selected_pairs": str(selected_path),
        "input_summary": str(input_summary_path),
        "oof_bucket_summary": str(oof_bucket_summary_path),
        "index_html": str(index_path),
        "good_bad_oof_index_html": str(good_bad_index_path),
        "global_local_index_html": str(global_local_index_path),
        "wrong_depth_index_html": str(wrong_depth_index_path),
        "figures_dir": str(figures_dir),
        "simple_overlay_dir": str(simple_overlay_dir),
        "global_local_dir": str(global_local_dir),
        "summary": str(summary_path),
    },
    "artifact_sha256": {
        "scored_pairs_gzip": sha256_file(scored_path),
        "scored_pairs_decompressed": sha256_decompressed(scored_path),
        "selected_pairs": sha256_file(selected_path),
        "input_summary": sha256_file(input_summary_path),
        "oof_bucket_summary": sha256_file(oof_bucket_summary_path),
        "index_html": sha256_file(index_path),
        "good_bad_oof_index_html": sha256_file(good_bad_index_path),
        "global_local_index_html": sha256_file(global_local_index_path),
        "wrong_depth_index_html": sha256_file(wrong_depth_index_path),
    },
    "oof_join": oof_summary,
    "input_files": {
        "example_horizontal_sha256": {
            path.name: sha256_file(path) for path in sample_horizontal
        }
    },
    "runtime_sec": float(time.time() - start_time),
    "audit_config": audit_config,
}
write_json(summary_path, summary)

print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
print("HTML index:", index_path)
print("Figures dir:", figures_dir)
