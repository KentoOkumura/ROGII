from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from settings import KAGGLE_INPUT_ROOT, ExperimentPaths, get_nested, load_config

FORMATION_COLUMNS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
RAW_PHYSICAL_COLUMNS = ["MD", "Z", *FORMATION_COLUMNS]
PHYSICAL_DERIVATIVE_COLUMNS = [
    "dTVT_dMD",
    "dZ_dMD",
    "neg_dZ_dMD",
    "dANCC_dMD",
    "dANCC_minus_Z_dMD",
]


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path
    well_column: str
    target_column: str
    target_delta_column: str | None
    base_column: str | None
    x_column: str
    cutoff_column: str | None
    selected_cutoff: float | None
    candidate_columns: list[dict[str, str]]
    confidence_columns: list[str]
    disagreement_columns: dict[str, str]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed else Path.open
    with opener(path, "rb") as fp:  # type: ignore[arg-type]
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_source_path(root: Path, source_config: dict[str, Any]) -> Path:
    local_path = source_config.get("local_path")
    if local_path:
        candidate = Path(str(local_path))
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.exists():
            return candidate

    slug = str(source_config.get("kaggle_source_slug") or "").strip()
    relative_paths = [str(item) for item in source_config.get("relative_paths", [])]
    if KAGGLE_INPUT_ROOT.exists():
        input_roots: list[Path] = []
        if slug:
            input_roots.append(KAGGLE_INPUT_ROOT / slug)
        input_roots.extend(path for path in sorted(KAGGLE_INPUT_ROOT.iterdir()) if path.is_dir())
        seen: set[Path] = set()
        for input_root in input_roots:
            if input_root in seen or not input_root.exists():
                continue
            seen.add(input_root)
            for relative_path in relative_paths:
                candidate = input_root / relative_path
                if candidate.exists():
                    return candidate
            for filename in (
                "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz",
                "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv",
            ):
                matches = sorted(input_root.rglob(filename))
                if matches:
                    return matches[0]

    raise FileNotFoundError(
        "PF/Beam source artifact not found. Checked local_path and Kaggle input source "
        f"for source={source_config.get('name')!r}."
    )


def build_source_spec(config: dict[str, Any], root: Path) -> SourceSpec:
    source_name = str(get_nested(config, "eda.source_name") or "")
    sources = get_nested(config, "data.pfbeam_sources") or []
    source_config = next((item for item in sources if item.get("name") == source_name), None)
    if source_config is None:
        raise ValueError(f"data.pfbeam_sources does not contain eda.source_name={source_name!r}")

    selected_cutoff = get_nested(config, "eda.selected_cutoff")
    if selected_cutoff is None:
        selected_cutoff = source_config.get("selected_cutoff")

    return SourceSpec(
        name=str(source_config["name"]),
        path=find_source_path(root, source_config),
        well_column=str(source_config.get("well_column", "well_id")),
        target_column=str(source_config.get("target_column", "target_tvt")),
        target_delta_column=source_config.get("target_delta_column"),
        base_column=source_config.get("base_column"),
        x_column=str(source_config.get("x_column", "MD")),
        cutoff_column=source_config.get("cutoff_column"),
        selected_cutoff=float(selected_cutoff) if selected_cutoff is not None else None,
        candidate_columns=list(source_config.get("candidate_columns", [])),
        confidence_columns=list(source_config.get("confidence_columns", [])),
        disagreement_columns=dict(source_config.get("disagreement_columns", {})),
    )


def materialize_tvt_columns(frame: pd.DataFrame, source: SourceSpec) -> pd.DataFrame:
    frame = frame.copy()
    if source.target_column not in frame:
        if not source.base_column or not source.target_delta_column:
            raise ValueError(
                f"{source.target_column!r} is missing and base/target delta "
                "columns are not configured."
            )
        missing = [
            col for col in (source.base_column, source.target_delta_column) if col not in frame
        ]
        if missing:
            raise ValueError(f"Cannot materialize true TVT; missing columns: {missing}")
        frame[source.target_column] = pd.to_numeric(
            frame[source.base_column], errors="coerce"
        ) + pd.to_numeric(frame[source.target_delta_column], errors="coerce")

    if source.base_column and source.base_column in frame:
        base = pd.to_numeric(frame[source.base_column], errors="coerce")
    else:
        base = None

    for spec in source.candidate_columns:
        name = spec.get("name")
        source_column = spec.get("source_column") or name
        transform = spec.get("transform", "absolute")
        if not name or not source_column or source_column not in frame:
            continue
        values = pd.to_numeric(frame[source_column], errors="coerce")
        if transform == "base_plus_delta":
            if base is None:
                raise ValueError(
                    f"Candidate {name!r} requires base_plus_delta but no base column is available."
                )
            frame[name] = base + values
        elif transform == "absolute":
            frame[name] = values
        else:
            raise ValueError(f"Unsupported candidate transform for {name!r}: {transform!r}")

    if {"pf_ancc", "beam_mean"}.issubset(frame.columns):
        diff = pd.to_numeric(frame["pf_ancc"], errors="coerce") - pd.to_numeric(
            frame["beam_mean"], errors="coerce"
        )
        frame["pf_ancc_vs_beam_mean"] = diff
        frame["pf_ancc_vs_beam_mean_abs"] = diff.abs()
    return frame


def read_source_frame(source: SourceSpec, *, debug_max_wells: int | None = None) -> pd.DataFrame:
    dtype = {"id": str, source.well_column: str}
    frame = pd.read_csv(source.path, dtype=dtype, low_memory=False)
    frame[source.well_column] = frame[source.well_column].astype(str)
    frame = materialize_tvt_columns(frame, source)
    if (
        source.cutoff_column
        and source.selected_cutoff is not None
        and source.cutoff_column in frame
    ):
        cutoff_values = pd.to_numeric(frame[source.cutoff_column], errors="coerce")
        frame = frame[np.isclose(cutoff_values, float(source.selected_cutoff), atol=1e-9)].copy()
    if debug_max_wells is not None and debug_max_wells > 0:
        wells = sorted(frame[source.well_column].dropna().astype(str).unique())[
            : int(debug_max_wells)
        ]
        frame = frame[frame[source.well_column].astype(str).isin(wells)].copy()
    return frame


def parse_raw_row_index(frame: pd.DataFrame) -> pd.Series:
    if "id" not in frame:
        raise ValueError("Source frame must contain id to join raw train physical columns.")
    suffix = frame["id"].astype(str).str.rsplit("_", n=1).str[-1]
    row_idx = pd.to_numeric(suffix, errors="coerce")
    missing = row_idx.isna()
    if bool(missing.any()):
        examples = frame.loc[missing, "id"].astype(str).head(5).tolist()
        raise ValueError(f"Could not parse raw row index from id examples: {examples}")
    return row_idx.astype(np.int64)


def load_raw_physical_context(
    train_dir: Path,
    wells: list[str],
    *,
    well_column: str,
    required_columns: list[str],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    usecols = sorted(set(required_columns))
    for well_id in wells:
        path = train_dir / f"{well_id}__horizontal_well.csv"
        if not path.exists():
            raise FileNotFoundError(f"Raw train horizontal well CSV not found: {path}")
        raw = pd.read_csv(path, usecols=lambda col: col in usecols)
        available = [col for col in usecols if col in raw]
        missing_required = [col for col in ("MD", "Z") if col not in raw]
        if missing_required:
            raise ValueError(f"{path} missing required physical columns: {missing_required}")
        raw = raw[available].copy()
        raw[well_column] = str(well_id)
        raw["raw_row_idx"] = np.arange(len(raw), dtype=np.int64)
        rows.append(raw)
    if not rows:
        return pd.DataFrame(columns=[well_column, "raw_row_idx", *usecols])
    return pd.concat(rows, ignore_index=True)


def _diff_per_md(values: pd.Series, md: pd.Series) -> pd.Series:
    values = pd.to_numeric(values, errors="coerce")
    md = pd.to_numeric(md, errors="coerce")
    denom = md.diff()
    denom = denom.where(denom.abs() > 1e-12)
    out = values.diff() / denom
    return out.replace([np.inf, -np.inf], np.nan)


def add_physical_derivatives(frame: pd.DataFrame, source: SourceSpec) -> pd.DataFrame:
    frame = frame.copy()
    frame["ANCC_minus_Z"] = pd.to_numeric(frame["ANCC"], errors="coerce") - pd.to_numeric(
        frame["Z"], errors="coerce"
    )
    for column in PHYSICAL_DERIVATIVE_COLUMNS:
        frame[column] = np.nan

    sort_column = "MD" if "MD" in frame else source.x_column
    for _, group in frame.groupby(source.well_column, sort=False):
        group = group.sort_values(sort_column if sort_column in group else source.x_column)
        idx = group.index
        md = group[sort_column] if sort_column in group else group[source.x_column]
        frame.loc[idx, "dTVT_dMD"] = _diff_per_md(group[source.target_column], md)
        dz_dmd = _diff_per_md(group["Z"], md)
        frame.loc[idx, "dZ_dMD"] = dz_dmd
        frame.loc[idx, "neg_dZ_dMD"] = -dz_dmd
        frame.loc[idx, "dANCC_dMD"] = _diff_per_md(group["ANCC"], md)
        frame.loc[idx, "dANCC_minus_Z_dMD"] = _diff_per_md(group["ANCC_minus_Z"], md)
    return frame


def attach_raw_physical_context(
    frame: pd.DataFrame,
    source: SourceSpec,
    train_dir: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    physical_cfg = get_nested(config, "eda.physical_decomposition") or {}
    requested_formations = list(physical_cfg.get("formation_columns") or FORMATION_COLUMNS)
    required_columns = ["MD", "Z", *requested_formations]
    frame = frame.copy()
    frame["raw_row_idx"] = parse_raw_row_index(frame)
    wells = sorted(frame[source.well_column].dropna().astype(str).unique().tolist())
    context = load_raw_physical_context(
        train_dir,
        wells,
        well_column=source.well_column,
        required_columns=required_columns,
    )
    before_rows = len(frame)
    frame = frame.merge(
        context,
        on=[source.well_column, "raw_row_idx"],
        how="left",
        validate="many_to_one",
    )
    if len(frame) != before_rows:
        raise ValueError("Raw physical context join changed row count.")

    coverage = {
        col: float(pd.to_numeric(frame[col], errors="coerce").notna().mean())
        for col in required_columns
        if col in frame
    }
    for col in ("MD", "Z"):
        if coverage.get(col, 0.0) <= 0.0:
            raise ValueError(f"Raw physical context join produced no finite values for {col}")

    frame = add_physical_derivatives(frame, source)
    derivative_coverage = {
        col: float(pd.to_numeric(frame[col], errors="coerce").notna().mean())
        for col in PHYSICAL_DERIVATIVE_COLUMNS
        if col in frame
    }
    return frame, {
        "enabled": True,
        "train_dir": str(train_dir),
        "raw_context_wells": int(len(wells)),
        "raw_context_rows": int(len(context)),
        "raw_context_columns": [col for col in required_columns if col in context],
        "coverage": coverage,
        "derivative_coverage": derivative_coverage,
    }


def rmse(values: pd.Series, target: pd.Series) -> float:
    delta = pd.to_numeric(values, errors="coerce") - pd.to_numeric(target, errors="coerce")
    delta = delta[np.isfinite(delta)]
    if delta.empty:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(delta.to_numpy(dtype=float)))))


def summarize_wells(frame: pd.DataFrame, source: SourceSpec) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    candidate_names = [
        item["name"] for item in source.candidate_columns if item.get("name") in frame
    ]
    target = source.target_column
    pf_col = next(
        (
            item["name"]
            for item in source.candidate_columns
            if item.get("role") == "selected_pf" and item.get("name") in candidate_names
        ),
        None,
    )
    beam_col = next(
        (
            item["name"]
            for item in source.candidate_columns
            if item.get("role") == "selected_beam" and item.get("name") in candidate_names
        ),
        None,
    )
    anchor_col = next(
        (
            item["name"]
            for item in source.candidate_columns
            if item.get("role") == "anchor" and item.get("name") in candidate_names
        ),
        None,
    )
    abs_disagreement_col = source.disagreement_columns.get("pf_beam_abs")

    for well_id, group in frame.groupby(source.well_column, sort=True):
        group = group.sort_values(source.x_column if source.x_column in group else "row_idx")
        row: dict[str, Any] = {
            "well_id": str(well_id),
            "rows": int(len(group)),
            "x_min": float(pd.to_numeric(group[source.x_column], errors="coerce").min())
            if source.x_column in group
            else None,
            "x_max": float(pd.to_numeric(group[source.x_column], errors="coerce").max())
            if source.x_column in group
            else None,
            "target_min": float(pd.to_numeric(group[target], errors="coerce").min()),
            "target_max": float(pd.to_numeric(group[target], errors="coerce").max()),
        }
        for column in (
            "prefix_length",
            "eval_step",
            "cutoff_row",
            "selector_n_eval",
            "selector_z_span",
        ):
            if column in group:
                values = pd.to_numeric(group[column], errors="coerce")
                row[f"{column}_median"] = float(values.median())
                row[f"{column}_max"] = float(values.max())
        for column in candidate_names:
            row[f"{column}_rmse"] = rmse(group[column], group[target])
            row[f"{column}_bias"] = float(
                (
                    pd.to_numeric(group[column], errors="coerce")
                    - pd.to_numeric(group[target], errors="coerce")
                ).mean()
            )
        if pf_col and beam_col:
            diff = pd.to_numeric(group[pf_col], errors="coerce") - pd.to_numeric(
                group[beam_col], errors="coerce"
            )
            row["pf_beam_diff_mean"] = float(diff.mean())
            row["pf_beam_abs_diff_mean"] = float(diff.abs().mean())
            row["pf_beam_abs_diff_p95"] = float(diff.abs().quantile(0.95))
            row["beam_minus_pf_rmse_delta"] = row.get(f"{beam_col}_rmse", math.nan) - row.get(
                f"{pf_col}_rmse", math.nan
            )
            row["primary_pf_column"] = pf_col
            row["primary_beam_column"] = beam_col
            row["primary_pf_rmse"] = row.get(f"{pf_col}_rmse", math.nan)
            row["primary_beam_rmse"] = row.get(f"{beam_col}_rmse", math.nan)
        if abs_disagreement_col and abs_disagreement_col in group:
            row["source_pf_beam_abs_diff_mean"] = float(
                pd.to_numeric(group[abs_disagreement_col], errors="coerce").mean()
            )
        if pf_col and anchor_col:
            row["anchor_minus_pf_rmse_delta"] = row.get(f"{anchor_col}_rmse", math.nan) - row.get(
                f"{pf_col}_rmse", math.nan
            )
            row["anchor_column"] = anchor_col
            row["anchor_rmse"] = row.get(f"{anchor_col}_rmse", math.nan)
        for column in source.confidence_columns:
            if column in group:
                values = pd.to_numeric(group[column], errors="coerce")
                row[f"{column}_mean"] = float(values.mean())
                row[f"{column}_p95"] = float(values.quantile(0.95))
        rows.append(row)
    return pd.DataFrame(rows)


def unique_append(
    selected: list[tuple[str, str]], reason: str, values: pd.Series, count: int, *, ascending: bool
) -> None:
    if count <= 0 or values.empty:
        return
    ordered = values.dropna().sort_values(ascending=ascending)
    for well_id in ordered.index.astype(str).tolist():
        if any(item[1] == well_id for item in selected):
            continue
        selected.append((reason, well_id))
        if sum(1 for item in selected if item[0] == reason) >= count:
            break


def select_representative_wells(well_summary: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    selected_wells = get_nested(config, "eda.selected_wells")
    if selected_wells:
        available = set(well_summary["well_id"].astype(str))
        rows = [
            {"reason": "selected_wells", "well_id": str(well_id)}
            for well_id in selected_wells
            if str(well_id) in available
        ]
        return pd.DataFrame(rows)

    if bool(get_nested(config, "eda.plot_all_wells")):
        wells = sorted(well_summary["well_id"].astype(str).tolist())
        return pd.DataFrame([{"reason": "all_wells", "well_id": well_id} for well_id in wells])

    counts = get_nested(config, "eda.representative_counts") or {}
    max_plots = get_nested(config, "eda.max_plots")
    max_plots = None if max_plots in {None, "null", ""} else int(max_plots)
    summary = well_summary.set_index("well_id", drop=False)
    selected: list[tuple[str, str]] = []

    if "primary_pf_rmse" in summary:
        unique_append(
            selected,
            "best_pf_rmse",
            summary["primary_pf_rmse"],
            int(counts.get("best_pf_rmse", 0)),
            ascending=True,
        )
        unique_append(
            selected,
            "worst_pf_rmse",
            summary["primary_pf_rmse"],
            int(counts.get("worst_pf_rmse", 0)),
            ascending=False,
        )
    disagreement_col = (
        "pf_beam_abs_diff_mean"
        if "pf_beam_abs_diff_mean" in summary
        else "source_pf_beam_abs_diff_mean"
    )
    if disagreement_col in summary:
        unique_append(
            selected,
            "highest_pf_beam_disagreement",
            summary[disagreement_col],
            int(counts.get("highest_pf_beam_disagreement", 0)),
            ascending=False,
        )
    if "beam_minus_pf_rmse_delta" in summary:
        unique_append(
            selected,
            "beam_beats_pf",
            -summary["beam_minus_pf_rmse_delta"],
            int(counts.get("beam_beats_pf", 0)),
            ascending=False,
        )
    if "anchor_minus_pf_rmse_delta" in summary:
        unique_append(
            selected,
            "anchor_beats_pf",
            -summary["anchor_minus_pf_rmse_delta"],
            int(counts.get("anchor_beats_pf", 0)),
            ascending=False,
        )
    unique_append(
        selected,
        "longest_eval_tail",
        summary["rows"],
        int(counts.get("longest_eval_tail", 0)),
        ascending=False,
    )

    random_count = int(counts.get("stable_random", 0))
    if random_count > 0:
        seed = int(get_nested(config, "reproducibility.seed") or 42)
        sampled = (
            summary.sample(n=min(random_count, len(summary)), random_state=seed)
            .index.astype(str)
            .tolist()
        )
        for well_id in sampled:
            if any(item[1] == well_id for item in selected):
                continue
            selected.append(("stable_random", well_id))

    if max_plots is not None:
        selected = selected[:max_plots]
    return pd.DataFrame([{"reason": reason, "well_id": well_id} for reason, well_id in selected])


def downsample_for_plot(group: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(group) <= max_points:
        return group
    indices = np.linspace(0, len(group) - 1, int(max_points)).round().astype(int)
    return group.iloc[np.unique(indices)].copy()


def add_prediction_start_line(
    ax: Any,
    x: pd.Series,
    config: dict[str, Any],
    *,
    with_label: bool,
) -> None:
    marker_cfg = get_nested(config, "eda.prediction_start_line") or {}
    if not bool(marker_cfg.get("enabled", False)):
        return

    x_value = float(marker_cfg.get("x_value", 0.0))
    finite_x = pd.to_numeric(x, errors="coerce")
    finite_x = finite_x[np.isfinite(finite_x)]
    if finite_x.empty:
        return
    x_min = float(finite_x.min())
    x_max = float(finite_x.max())
    if x_value < x_min:
        if not bool(marker_cfg.get("fallback_to_first_x", True)):
            return
        x_value = x_min
    elif x_value > x_max:
        return

    ax.axvline(
        x_value,
        color=str(marker_cfg.get("color") or "#dc2626"),
        linestyle=str(marker_cfg.get("linestyle") or ":"),
        linewidth=float(marker_cfg.get("linewidth", 1.5)),
        alpha=float(marker_cfg.get("alpha", 0.9)),
        label=str(marker_cfg.get("label") or "prediction start") if with_label else None,
        zorder=4,
    )


def _raw_row_indices(frame: pd.DataFrame) -> np.ndarray:
    return parse_raw_row_index(frame).to_numpy(np.int64)


def candidate_abs_from_replay(frame: pd.DataFrame, candidate: str) -> np.ndarray | None:
    if "last_known_tvt" in frame:
        last = pd.to_numeric(frame["last_known_tvt"], errors="coerce").to_numpy(np.float32)
    else:
        last = None
    if candidate == "pf_ancc" and "pf_ancc" in frame:
        return pd.to_numeric(frame["pf_ancc"], errors="coerce").to_numpy(np.float32)
    if candidate == "pf_z" and "pf_z" in frame:
        return pd.to_numeric(frame["pf_z"], errors="coerce").to_numpy(np.float32)
    if candidate == "beam_mean" and "beam_mean_d" in frame and last is not None:
        return last + pd.to_numeric(frame["beam_mean_d"], errors="coerce").to_numpy(np.float32)
    if candidate == "likpf_mean" and "likpf_mean" in frame:
        return pd.to_numeric(frame["likpf_mean"], errors="coerce").to_numpy(np.float32)
    return None


def build_known_prefix_replay_overlay(
    wells: list[str],
    paths: ExperimentPaths,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    overlay_cfg = get_nested(config, "eda.known_replay_overlay") or {}
    if not bool(overlay_cfg.get("enabled")) or not wells:
        return pd.DataFrame(), {"enabled": False}

    try:
        from public_notebook_replay_audit import (
            build_well,
            configure_public_runtime,
            init_imputers,
            lik_pf,
            stable_seed,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "known_replay_overlay.enabled requires public_notebook_replay_audit.py in the "
            "experiment package."
        ) from exc

    holdout_rows = int(overlay_cfg.get("prefix_holdout_rows", 256))
    min_known = int(overlay_cfg.get("min_known_prefix_rows", 80))
    min_cal = int(overlay_cfg.get("min_calibration_rows", 32))
    use_likpf = bool(overlay_cfg.get("use_likpf", True))
    candidate_names = list(overlay_cfg.get("candidate_names") or ["pf_ancc", "pf_z", "beam_mean", "likpf_mean"])

    configure_public_runtime(
        data_dir=paths.raw_data_dir,
        output_dir=paths.artifacts_dir,
        n_jobs=int(overlay_cfg.get("n_jobs", 2)),
        pf_seeds=int(overlay_cfg.get("pf_seeds", 32)),
        pf_particles=int(overlay_cfg.get("pf_particles", 300)),
    )
    train_wells = sorted(
        path.stem.replace("__horizontal_well", "")
        for path in paths.train_data_dir.glob("*__horizontal_well.csv")
    )
    init_imputers(train_wells)

    overlay_frames: list[pd.DataFrame] = []
    status_rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="exp083_known_replay_overlay_") as tmp:
        temp_root = Path(tmp)
        for well_id in wells:
            hw_path = paths.train_data_dir / f"{well_id}__horizontal_well.csv"
            tw_path = paths.train_data_dir / f"{well_id}__typewell.csv"
            if not hw_path.exists() or not tw_path.exists():
                status_rows.append({"well_id": well_id, "status": "missing_raw_files"})
                continue
            horizontal = pd.read_csv(hw_path, low_memory=False)
            known = np.flatnonzero(
                pd.to_numeric(horizontal["TVT_input"], errors="coerce").notna().to_numpy()
            )
            if len(known) < min_known + min_cal:
                status_rows.append(
                    {"well_id": well_id, "status": "too_short_prefix", "known_rows": int(len(known))}
                )
                continue
            start = max(min_known, len(known) - holdout_rows)
            holdout_idx = known[start:]
            if len(holdout_idx) < min_cal:
                status_rows.append(
                    {
                        "well_id": well_id,
                        "status": "too_few_holdout_rows",
                        "holdout_rows": int(len(holdout_idx)),
                    }
                )
                continue

            masked = horizontal.copy()
            masked.loc[holdout_idx, "TVT_input"] = np.nan
            temp_hw = temp_root / f"{well_id}__horizontal_well.csv"
            masked.to_csv(temp_hw, index=False)

            replay = build_well(temp_hw, tw_path, is_train=True)
            if replay is None or len(replay) == 0:
                status_rows.append(
                    {"well_id": well_id, "status": "replay_failed", "holdout_rows": int(len(holdout_idx))}
                )
                continue
            replay = replay.copy()
            replay["row_idx"] = _raw_row_indices(replay)

            if use_likpf:
                typewell = pd.read_csv(tw_path).sort_values("TVT")
                out, ev_index, _ = lik_pf(
                    masked,
                    typewell,
                    seed_base=stable_seed("exp083_known_replay_overlay", well_id),
                )
                if len(ev_index) == len(replay) and "pf_mean" in out:
                    replay["likpf_mean"] = np.asarray(out["pf_mean"], dtype=np.float32)

            holdout_set = set(int(v) for v in holdout_idx)
            replay = replay[replay["row_idx"].isin(holdout_set)].reset_index(drop=True)
            if replay.empty:
                status_rows.append({"well_id": well_id, "status": "no_holdout_rows_after_replay"})
                continue

            anchor_idx = int(known[start - 1])
            anchor_md = float(horizontal.loc[anchor_idx, "MD"])
            out_frame = pd.DataFrame(
                {
                    "well_id": str(well_id),
                    "row_idx": replay["row_idx"].astype(np.int64),
                    "known_replay_anchor_row_idx": anchor_idx,
                    "known_replay_anchor_md": anchor_md,
                    "known_replay_md_since": pd.to_numeric(
                        horizontal.loc[replay["row_idx"].to_numpy(np.int64), "MD"],
                        errors="coerce",
                    ).to_numpy(np.float32)
                    - np.float32(anchor_md),
                }
            )
            for candidate in candidate_names:
                values = candidate_abs_from_replay(replay, candidate)
                if values is not None:
                    out_frame[f"known_replay_{candidate}"] = values.astype(np.float32)
            overlay_frames.append(out_frame)
            status_rows.append(
                {
                    "well_id": well_id,
                    "status": "ok",
                    "known_rows": int(len(known)),
                    "holdout_rows": int(len(holdout_idx)),
                    "replay_rows": int(len(out_frame)),
                    "anchor_row_idx": anchor_idx,
                }
            )

    overlay = pd.concat(overlay_frames, ignore_index=True) if overlay_frames else pd.DataFrame()
    status = pd.DataFrame(status_rows)
    return overlay, {
        "enabled": True,
        "mode": "exp169_prefix_holdout_replay",
        "requested_wells": wells,
        "rows": int(len(overlay)),
        "ok_wells": int(status["status"].eq("ok").sum()) if "status" in status else 0,
        "status_counts": status["status"].value_counts().to_dict() if "status" in status else {},
        "prefix_holdout_rows": holdout_rows,
        "min_known_prefix_rows": min_known,
        "status_rows": status.to_dict(orient="records"),
    }


def append_known_prefix_and_overlay(
    group: pd.DataFrame,
    source: SourceSpec,
    train_dir: Path,
    overlay: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    overlay_cfg = get_nested(config, "eda.known_replay_overlay") or {}
    if not bool(overlay_cfg.get("enabled")):
        return group, {"enabled": False}

    well_id = str(group[source.well_column].dropna().astype(str).iloc[0])
    hw_path = train_dir / f"{well_id}__horizontal_well.csv"
    if not hw_path.exists():
        return group, {"enabled": True, "status": "missing_raw_file"}

    horizontal = pd.read_csv(hw_path, low_memory=False)
    known_idx = np.flatnonzero(
        pd.to_numeric(horizontal["TVT_input"], errors="coerce").notna().to_numpy()
    )
    if len(known_idx) == 0:
        return group, {"enabled": True, "status": "no_known_tvt_input"}

    tail_indices = _raw_row_indices(group)
    first_tail_idx = int(np.nanmin(tail_indices)) if len(tail_indices) else int(known_idx[-1] + 1)
    known_before_tail = known_idx[known_idx < first_tail_idx]
    anchor_idx = int(known_before_tail[-1]) if len(known_before_tail) else int(known_idx[-1])
    anchor_md = float(horizontal.loc[anchor_idx, "MD"])

    prefix_idx = known_idx[known_idx <= anchor_idx]
    existing = set(int(v) for v in tail_indices)
    prefix_idx = np.asarray([int(v) for v in prefix_idx if int(v) not in existing], dtype=np.int64)

    group = group.copy()
    group["row_idx"] = tail_indices
    prefix = pd.DataFrame(columns=group.columns)
    if len(prefix_idx):
        raw_prefix = horizontal.loc[prefix_idx].copy()
        prefix = pd.DataFrame(index=np.arange(len(raw_prefix)), columns=group.columns)
        prefix[source.well_column] = well_id
        prefix["id"] = [f"{well_id}_{int(idx)}" for idx in prefix_idx]
        prefix["row_idx"] = prefix_idx
        prefix[source.x_column] = pd.to_numeric(raw_prefix["MD"], errors="coerce").to_numpy(np.float32) - np.float32(anchor_md)
        prefix[source.target_column] = pd.to_numeric(raw_prefix["TVT"], errors="coerce").to_numpy(np.float32)

    combined = pd.concat([prefix, group], ignore_index=True, sort=False)
    if not overlay.empty:
        well_overlay = overlay[overlay["well_id"].astype(str).eq(well_id)].copy()
        if not well_overlay.empty:
            combined = combined.merge(
                well_overlay.drop(columns=["well_id"]),
                on="row_idx",
                how="left",
                validate="many_to_one",
            )
    combined = combined.sort_values(source.x_column).reset_index(drop=True)
    overlay_cols = [col for col in combined.columns if col.startswith("known_replay_")]
    return combined, {
        "enabled": True,
        "status": "ok",
        "known_prefix_rows_added": int(len(prefix)),
        "anchor_row_idx": anchor_idx,
        "anchor_md": anchor_md,
        "overlay_rows": int(combined[overlay_cols].notna().any(axis=1).sum()) if overlay_cols else 0,
    }


def plot_well(
    group: pd.DataFrame,
    source: SourceSpec,
    config: dict[str, Any],
    well_metrics: dict[str, Any],
    reason: str,
    output_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to write PNG plots. Run this notebook on Kaggle "
            "or install matplotlib in the local debug environment."
        ) from exc

    plot_cfg = get_nested(config, "eda.plot_columns") or {}
    primary = [col for col in plot_cfg.get("primary", []) if col in group]
    secondary = [col for col in plot_cfg.get("secondary", []) if col in group]
    max_points = int(get_nested(config, "eda.max_points_per_plot") or 2500)
    x_column = source.x_column if source.x_column in group else "row_idx"
    group = downsample_for_plot(group.sort_values(x_column), max_points)
    x = pd.to_numeric(group[x_column], errors="coerce")
    target = pd.to_numeric(group[source.target_column], errors="coerce")

    derivative_cfg = get_nested(config, "eda.derivative_panel") or {}
    use_derivative_panel = bool(derivative_cfg.get("enabled"))
    if use_derivative_panel:
        fig, (ax, ax_deriv) = plt.subplots(
            2,
            1,
            figsize=(12.5, 8.3),
            dpi=140,
            sharex=True,
            gridspec_kw={"height_ratios": [3.2, 1.0]},
        )
    else:
        fig, ax = plt.subplots(figsize=(12.5, 6.8), dpi=140)
        ax_deriv = None

    background_cfg = get_nested(config, "eda.physical_background") or {}
    if bool(background_cfg.get("enabled")):
        target_finite = target[np.isfinite(target)]
        if not target_finite.empty:
            y_low = float(target_finite.quantile(float(background_cfg.get("target_low_q", 0.02))))
            y_high = float(
                target_finite.quantile(float(background_cfg.get("target_high_q", 0.98)))
            )
            if not np.isfinite(y_low) or not np.isfinite(y_high) or y_low == y_high:
                y_low = float(target_finite.min())
                y_high = float(target_finite.max())

            scaled_background: dict[str, pd.Series] = {}
            background_columns = list(background_cfg.get("columns", []))
            common_scale = bool(background_cfg.get("common_scale", False))
            common_values: list[pd.Series] = []
            if common_scale:
                for item in background_columns:
                    column = item.get("name")
                    if not column or column not in group:
                        continue
                    if not bool(item.get("use_common_scale", True)):
                        continue
                    values = pd.to_numeric(group[column], errors="coerce")
                    if str(item.get("transform", "raw")) == "negate":
                        values = -values
                    finite = values[np.isfinite(values)]
                    if not finite.empty:
                        common_values.append(finite)
                common_finite = (
                    pd.concat(common_values, ignore_index=True)
                    if common_values
                    else pd.Series(dtype=float)
                )
                common_low = (
                    float(
                        common_finite.quantile(float(background_cfg.get("common_low_q", 0.02)))
                    )
                    if not common_finite.empty
                    else float("nan")
                )
                common_high = (
                    float(
                        common_finite.quantile(float(background_cfg.get("common_high_q", 0.98)))
                    )
                    if not common_finite.empty
                    else float("nan")
                )
            else:
                common_low = float("nan")
                common_high = float("nan")

            for item in background_columns:
                column = item.get("name")
                if not column or column not in group:
                    continue
                values = pd.to_numeric(group[column], errors="coerce")
                transform = str(item.get("transform", "raw"))
                if transform == "negate":
                    values = -values
                finite = values[np.isfinite(values)]
                if finite.empty:
                    continue
                if common_scale and bool(item.get("use_common_scale", True)):
                    v_low = common_low
                    v_high = common_high
                else:
                    v_low = float(finite.quantile(float(item.get("low_q", 0.02))))
                    v_high = float(finite.quantile(float(item.get("high_q", 0.98))))
                if not np.isfinite(v_low) or not np.isfinite(v_high) or v_low == v_high:
                    continue
                scaled = y_low + (values - v_low) * (y_high - y_low) / (v_high - v_low)
                scaled_background[column] = scaled

            for band in background_cfg.get("bands", []):
                upper = str(band.get("upper") or "")
                lower = str(band.get("lower") or "")
                if upper not in scaled_background or lower not in scaled_background:
                    continue
                ax.fill_between(
                    x,
                    scaled_background[upper],
                    scaled_background[lower],
                    color=band.get("color"),
                    alpha=float(band.get("alpha", 0.08)),
                    linewidth=0,
                    label=str(band.get("label", f"{upper}-{lower} band")),
                    zorder=0,
                )

            for item in background_columns:
                column = item.get("name")
                if not column or column not in scaled_background:
                    continue
                scaled = scaled_background[column]
                ax.plot(
                    x,
                    scaled,
                    linewidth=float(item.get("linewidth", 1.0)),
                    alpha=float(item.get("alpha", 0.18)),
                    linestyle=str(item.get("linestyle", "-")),
                    color=item.get("color"),
                    label=str(item.get("label", f"{column} scaled")),
                    zorder=1,
                )

    ax.plot(x, target, color="black", linewidth=2.2, label="true TVT", zorder=5)
    add_prediction_start_line(ax, x, config, with_label=True)

    colors = {
        "last_anchor_tvt": "#777777",
        "pf_ancc": "#1f77b4",
        "pf_z": "#9467bd",
        "beam_mean": "#ff7f0e",
        "likpf_mean": "#2ca02c",
        "pf_pred": "#1f77b4",
        "pf_selected_scale_pred": "#2ca02c",
        "beam_pred": "#ff7f0e",
    }
    linestyles = {
        "last_anchor_tvt": "--",
        "pf_ancc": "-",
        "pf_z": "-",
        "beam_mean": "-",
        "likpf_mean": "-",
    }
    labels = {
        item.get("name"): item.get("label", item.get("name"))
        for item in source.candidate_columns
        if item.get("name")
    }
    for column in secondary:
        ax.plot(
            x,
            pd.to_numeric(group[column], errors="coerce"),
            linewidth=0.85,
            alpha=0.35,
            color="#86a6c8",
            label=labels.get(column, column),
        )
    for column in primary:
        if column == source.target_column:
            continue
        ax.plot(
            x,
            pd.to_numeric(group[column], errors="coerce"),
            linewidth=1.8 if column in {"pf_ancc", "pf_z", "beam_mean", "likpf_mean"} else 1.25,
            linestyle=linestyles.get(column, "-"),
            alpha=0.9,
            color=colors.get(column),
            label=labels.get(column, column),
        )

    overlay_cfg = get_nested(config, "eda.known_replay_overlay") or {}
    if bool(overlay_cfg.get("enabled")):
        overlay_styles = {
            "known_replay_pf_ancc": ("known replay PF ANCC", "#1f77b4"),
            "known_replay_pf_z": ("known replay PF Z", "#9467bd"),
            "known_replay_beam_mean": ("known replay Beam mean", "#ff7f0e"),
            "known_replay_likpf_mean": ("known replay Likelihood PF mean", "#2ca02c"),
        }
        for column in overlay_cfg.get("plot_columns") or list(overlay_styles):
            if column not in group:
                continue
            values = pd.to_numeric(group[column], errors="coerce")
            if not bool(np.isfinite(values).any()):
                continue
            label, color = overlay_styles.get(column, (column, "#111827"))
            ax.plot(
                x,
                values,
                linewidth=float(overlay_cfg.get("linewidth", 2.0)),
                linestyle=str(overlay_cfg.get("linestyle", "--")),
                alpha=float(overlay_cfg.get("alpha", 0.95)),
                color=color,
                label=label,
                zorder=6,
            )

    title_parts = [
        f"well={well_metrics['well_id']}",
        f"reason={reason}",
        f"rows={well_metrics.get('rows')}",
    ]
    for column in ("primary_pf_rmse", "primary_beam_rmse", "anchor_rmse", "pf_beam_abs_diff_mean"):
        value = well_metrics.get(column)
        if value is not None and np.isfinite(value):
            title_parts.append(f"{column.replace('_rmse', '')}={float(value):.3f}")
    ax.set_title(" | ".join(title_parts), fontsize=10)
    ax.set_ylabel("TVT")
    ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.7)
    ax.legend(loc="best", fontsize=8, ncol=2)

    if ax_deriv is not None:
        derivative_col = str(derivative_cfg.get("column") or "dZ_dMD")
        derivative_label = str(derivative_cfg.get("label") or "dZ/dMD")
        if derivative_col in group:
            derivative_values = pd.to_numeric(group[derivative_col], errors="coerce")
            ax_deriv.plot(
                x,
                derivative_values,
                color=str(derivative_cfg.get("color") or "#0f172a"),
                linewidth=float(derivative_cfg.get("linewidth", 1.1)),
                alpha=float(derivative_cfg.get("alpha", 0.85)),
                label=derivative_label,
            )
            finite = derivative_values[np.isfinite(derivative_values)]
            if not finite.empty:
                clip_q = float(derivative_cfg.get("clip_quantile", 0.995))
                max_abs = float(finite.abs().quantile(clip_q))
                if np.isfinite(max_abs) and max_abs > 0:
                    ax_deriv.set_ylim(-max_abs * 1.12, max_abs * 1.12)
        ax_deriv.axhline(0.0, color="#9ca3af", linewidth=0.8, alpha=0.75)
        add_prediction_start_line(ax_deriv, x, config, with_label=False)
        ax_deriv.set_ylabel(derivative_label)
        ax_deriv.set_xlabel(x_column)
        ax_deriv.grid(True, color="#dddddd", linewidth=0.6, alpha=0.7)
        ax_deriv.legend(loc="best", fontsize=8)
    else:
        ax.set_xlabel(x_column)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _finite_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    values = []
    for column in columns:
        if column in frame:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            values.append(numeric[np.isfinite(numeric)])
    if not values:
        return pd.Series(dtype=float)
    return pd.concat(values, ignore_index=True)


def plot_physical_decomposition_well(
    group: pd.DataFrame,
    source: SourceSpec,
    config: dict[str, Any],
    well_metrics: dict[str, Any],
    reason: str,
    output_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to write PNG plots. Run this notebook on Kaggle "
            "or install matplotlib in the local debug environment."
        ) from exc

    physical_cfg = get_nested(config, "eda.physical_decomposition") or {}
    max_points = int(get_nested(config, "eda.max_points_per_plot") or 2500)
    x_column = str(physical_cfg.get("x_column") or "MD")
    if x_column not in group:
        x_column = source.x_column if source.x_column in group else "row_idx"
    group = downsample_for_plot(group.sort_values(x_column), max_points)
    x = pd.to_numeric(group[x_column], errors="coerce")

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13.5, 10.2),
        dpi=int(physical_cfg.get("dpi", 110)),
        sharex=True,
        gridspec_kw={"height_ratios": [1.05, 1.15, 1.0]},
    )
    ax_tvt, ax_depth, ax_slope = axes

    target = pd.to_numeric(group[source.target_column], errors="coerce")
    ax_tvt.plot(x, target, color="black", linewidth=2.1, label="true TVT", zorder=5)
    if "pf_z" in group:
        ax_tvt.plot(
            x,
            pd.to_numeric(group["pf_z"], errors="coerce"),
            color="#1f77b4",
            linewidth=1.7,
            alpha=0.9,
            label="PF Z",
        )
    if "last_anchor_tvt" in group:
        ax_tvt.plot(
            x,
            pd.to_numeric(group["last_anchor_tvt"], errors="coerce"),
            color="#777777",
            linewidth=1.1,
            linestyle="--",
            alpha=0.75,
            label="last anchor",
        )
    ax_tvt.set_ylabel("TVT")
    ax_tvt.legend(loc="best", fontsize=8, ncol=3)

    if "Z" in group:
        ax_depth.plot(
            x,
            pd.to_numeric(group["Z"], errors="coerce"),
            color="#111827",
            linewidth=1.55,
            label="Z",
        )
    formation_colors = {
        "ANCC": "#15803d",
        "ASTNU": "#9ca3af",
        "ASTNL": "#a3a3a3",
        "EGFDU": "#b8b8b8",
        "EGFDL": "#c6c6c6",
        "BUDA": "#d4d4d4",
    }
    for column in physical_cfg.get("formation_columns") or FORMATION_COLUMNS:
        if column not in group:
            continue
        ax_depth.plot(
            x,
            pd.to_numeric(group[column], errors="coerce"),
            color=formation_colors.get(column, "#b8b8b8"),
            linewidth=1.6 if column == "ANCC" else 0.85,
            alpha=0.9 if column == "ANCC" else 0.42,
            label=column,
        )
    ax_depth.set_ylabel("Z / formation top")
    ax_depth.legend(loc="best", fontsize=8, ncol=4)

    slope_styles = {
        "dTVT_dMD": ("dTVT/dMD", "black", 1.7, "-"),
        "neg_dZ_dMD": ("-dZ/dMD", "#2563eb", 1.45, "-"),
        "dANCC_dMD": ("dANCC/dMD", "#15803d", 1.35, "-"),
        "dANCC_minus_Z_dMD": ("d(ANCC - Z)/dMD", "#c2410c", 1.45, "-"),
    }
    for column, (label, color, width, linestyle) in slope_styles.items():
        if column not in group:
            continue
        ax_slope.plot(
            x,
            pd.to_numeric(group[column], errors="coerce"),
            color=color,
            linewidth=width,
            linestyle=linestyle,
            alpha=0.9,
            label=label,
        )
    ax_slope.axhline(0.0, color="#9ca3af", linewidth=0.8, alpha=0.75)
    slope_values = _finite_numeric(group, list(slope_styles))
    clip_q = float(physical_cfg.get("derivative_clip_quantile", 0.995))
    if not slope_values.empty:
        max_abs = float(slope_values.abs().quantile(clip_q))
        if np.isfinite(max_abs) and max_abs > 0:
            ax_slope.set_ylim(-max_abs * 1.12, max_abs * 1.12)
    ax_slope.set_ylabel("slope per MD")
    ax_slope.legend(loc="best", fontsize=8, ncol=2)

    jump_col = str(physical_cfg.get("jump_column") or "dTVT_dMD")
    if jump_col in group:
        jumps = pd.to_numeric(group[jump_col], errors="coerce").abs()
        jumps = jumps[np.isfinite(jumps)]
        if not jumps.empty:
            threshold = float(jumps.quantile(float(physical_cfg.get("jump_quantile", 0.99))))
            max_markers = int(physical_cfg.get("max_jump_markers", 24))
            marker_indices = jumps[jumps >= threshold].sort_values(ascending=False).head(
                max_markers
            ).index
            for jump_x in x.loc[marker_indices].dropna().tolist():
                for ax in axes:
                    ax.axvline(jump_x, color="#dc2626", linewidth=0.75, alpha=0.25)

    title_parts = [
        f"well={well_metrics['well_id']}",
        f"reason={reason}",
        f"rows={well_metrics.get('rows')}",
    ]
    for column in ("target_min", "target_max", "primary_pf_rmse"):
        value = well_metrics.get(column)
        if value is not None and np.isfinite(value):
            title_parts.append(f"{column}={float(value):.3f}")
    ax_tvt.set_title(" | ".join(title_parts), fontsize=10)
    for ax in axes:
        ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.7)
    ax_slope.set_xlabel(x_column)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def run_eda(
    *,
    config: dict[str, Any],
    paths: ExperimentPaths,
    debug: bool = False,
    max_plots: int | None = None,
) -> dict[str, Any]:
    paths.ensure_output_dirs()
    output_prefix = str(get_nested(config, "eda.output_prefix") or "pf_beam_true_tvt_2d_well_eda")
    plots_dir = paths.artifacts_dir / f"{output_prefix}_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    source = build_source_spec(config, paths.root)
    debug_max_wells = int(get_nested(config, "eda.debug_max_wells") or 6) if debug else None
    frame = read_source_frame(source, debug_max_wells=debug_max_wells)
    if frame.empty:
        raise ValueError("Source frame is empty after cutoff/debug filtering.")

    physical_summary: dict[str, Any] | None = None
    use_physical_decomposition = bool(
        get_nested(config, "eda.physical_decomposition.enabled")
    )
    use_physical_background = bool(get_nested(config, "eda.physical_background.enabled"))
    if use_physical_decomposition or use_physical_background:
        physical_summary = {
            "enabled": True,
            "join_scope": "per_plot_well",
            "train_dir": str(paths.train_data_dir),
            "note": (
                "Raw train physical columns are joined one selected well at a time to avoid "
                "materializing ANCC/Z context for the full exp072 cache."
            ),
        }

    required_columns = {source.well_column, source.target_column}
    missing = sorted(required_columns - set(frame.columns))
    if missing:
        raise ValueError(f"Source frame missing required columns: {missing}")

    well_summary = summarize_wells(frame, source)
    selected = select_representative_wells(well_summary, config)
    if max_plots is not None:
        selected = selected.head(int(max_plots)).copy()

    selected_wells = selected["well_id"].astype(str).tolist() if not selected.empty else []
    known_overlay, known_overlay_summary = build_known_prefix_replay_overlay(
        selected_wells,
        paths,
        config,
    )

    summary_by_well = well_summary.set_index("well_id", drop=False).to_dict(orient="index")
    plot_rows: list[dict[str, Any]] = []
    for _, selected_row in selected.iterrows():
        well_id = str(selected_row["well_id"])
        reason = str(selected_row["reason"])
        group = frame[frame[source.well_column].astype(str) == well_id]
        if group.empty:
            continue
        output_path = plots_dir / f"{reason}__{well_id}.png"
        if use_physical_decomposition:
            group, known_context_summary = append_known_prefix_and_overlay(
                group,
                source,
                paths.train_data_dir,
                known_overlay,
                config,
            )
            group, group_physical_summary = attach_raw_physical_context(
                group,
                source,
                paths.train_data_dir,
                config,
            )
            plot_physical_decomposition_well(
                group, source, config, summary_by_well[well_id], reason, output_path
            )
            plot_rows_extra = {
                "physical_context": group_physical_summary,
                "known_replay_overlay": known_context_summary,
            }
        else:
            group, known_context_summary = append_known_prefix_and_overlay(
                group,
                source,
                paths.train_data_dir,
                known_overlay,
                config,
            )
            if use_physical_background:
                group, group_physical_summary = attach_raw_physical_context(
                    group,
                    source,
                    paths.train_data_dir,
                    config,
                )
                plot_rows_extra = {
                    "physical_context": group_physical_summary,
                    "known_replay_overlay": known_context_summary,
                }
            else:
                plot_rows_extra = {"known_replay_overlay": known_context_summary}
            plot_well(group, source, config, summary_by_well[well_id], reason, output_path)
        plot_rows.append(
            {
                "well_id": well_id,
                "reason": reason,
                "plot_path": str(output_path.relative_to(paths.experiment_dir)),
                **summary_by_well[well_id],
                **plot_rows_extra,
            }
        )

    manifest = pd.DataFrame(plot_rows)
    well_summary_path = paths.artifacts_dir / f"{output_prefix}_well_summary.csv"
    manifest_path = paths.artifacts_dir / f"{output_prefix}_plot_manifest.csv"
    summary_path = paths.artifacts_dir / f"{output_prefix}_summary.json"
    zip_path = paths.artifacts_dir / f"{output_prefix}_plots.zip"

    well_summary.to_csv(well_summary_path, index=False)
    manifest.to_csv(manifest_path, index=False)

    if bool(get_nested(config, "eda.zip_plots")):
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for png_path in sorted(plots_dir.glob("*.png")):
                zf.write(png_path, arcname=png_path.name)

    compressed = source.path.suffix == ".gz"
    summary = {
        "experiment": get_nested(config, "experiment.name"),
        "status": "debug_completed" if debug else "eda_completed",
        "created_at": datetime.now(UTC).isoformat(),
        "debug": debug,
        "source": {
            "name": source.name,
            "path": str(source.path),
            "sha256": sha256_path(source.path),
            "decompressed_sha256": sha256_path(source.path, decompressed=True)
            if compressed
            else None,
            "rows_after_filter": int(len(frame)),
            "wells_after_filter": int(frame[source.well_column].nunique()),
            "selected_cutoff": source.selected_cutoff,
        },
        "outputs": {
            "well_summary_csv": str(well_summary_path),
            "plot_manifest_csv": str(manifest_path),
            "summary_json": str(summary_path),
            "plots_dir": str(plots_dir),
            "plots_zip": str(zip_path) if zip_path.exists() else None,
            "plot_count": int(len(manifest)),
        },
        "physical_context": physical_summary,
        "known_replay_overlay": known_overlay_summary,
        "metrics": {
            "primary_pf_rmse_mean_by_well": float(well_summary["primary_pf_rmse"].mean())
            if "primary_pf_rmse" in well_summary
            else None,
            "primary_beam_rmse_mean_by_well": float(well_summary["primary_beam_rmse"].mean())
            if "primary_beam_rmse" in well_summary
            else None,
            "anchor_rmse_mean_by_well": float(well_summary["anchor_rmse"].mean())
            if "anchor_rmse" in well_summary
            else None,
        },
    }
    summary_path.write_text(json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n")
    shutil.copyfile(summary_path, paths.metrics_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-local", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max-plots", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ExperimentPaths()
    if not args.allow_local:
        paths.require_kaggle_runtime()
    summary = run_eda(config=load_config(), paths=paths, debug=args.debug, max_plots=args.max_plots)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
