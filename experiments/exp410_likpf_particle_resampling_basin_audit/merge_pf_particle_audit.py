"""Strictly merge exp410 full shards and produce cross-angle PF cause readouts."""

from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


EXPERIMENT = "exp410_likpf_particle_resampling_basin_audit"
PREFIX = EXPERIMENT
DEFAULT_OUTPUT = Path(
    "experiments/exp410_likpf_particle_resampling_basin_audit/artifacts/full_merged"
)
DEFAULT_EPISODES = Path(
    "experiments/exp410_likpf_particle_resampling_basin_audit/assets/"
    "pf_persistent_offset_episodes.csv"
)
DEFAULT_WELLS = Path(
    "experiments/exp410_likpf_particle_resampling_basin_audit/assets/"
    "pf_target_wells.csv"
)
DEFAULT_SOURCE_ROWS = Path(
    "experiments/exp235_fixed_lag_particle_smoother_pf/artifacts/"
    "lag64_merged_v3/"
    "exp235_fixed_lag_particle_smoother_pf_merged_row_candidates.csv.gz"
)
DEFAULT_FULL_PACKAGE_CODE = Path(
    "experiments/exp410_likpf_particle_resampling_basin_audit/kaggle/"
    "train_variant0/"
    "exp410_likpf_particle_resampling_basin_audit_compact_selfcontained_train.py"
)
DEFAULT_FULL_PACKAGE_CONFIG = Path(
    "experiments/exp410_likpf_particle_resampling_basin_audit/kaggle/"
    "train_variant0/config.yaml"
)
EXPECTED_WELLS = 496
EXPECTED_SUFFIX_ROWS = 2_500_744
EXPECTED_EPISODES = 839
EXPECTED_EPISODE_ROWS = 819_288
EXPECTED_EPISODE_SSE = 453_149_095.6093302
EXPECTED_SOURCE_ROWS = 3_783_989
EXPECTED_SOURCE_WELLS = 773
EXPECTED_SOURCE_SHA256 = (
    "b9f6e3aab91478410dbba9f3779b3e9e90421641516892f9f451088c2c89c0bf"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", action="append", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--target-wells", type=Path, default=DEFAULT_WELLS)
    parser.add_argument("--source-rows", type=Path, default=DEFAULT_SOURCE_ROWS)
    parser.add_argument(
        "--full-package-code", type=Path, default=DEFAULT_FULL_PACKAGE_CODE
    )
    parser.add_argument(
        "--full-package-config", type=Path, default=DEFAULT_FULL_PACKAGE_CONFIG
    )
    parser.add_argument(
        "--write-large-ledgers",
        action="store_true",
        help=(
            "Write redundant merged row ledgers in addition to the four "
            "SHA-guarded canonical shard ledgers."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_path(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed else open
    with opener(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def require_artifact(root: Path, filename: str) -> Path:
    candidates = (
        root / "artifacts" / filename,
        root / filename,
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    matches = sorted(root.glob(f"**/{filename}"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"{root}: {filename}")


def cause_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    total_sse = float(episodes["episode_sse"].sum())
    result = (
        episodes.groupby("cause", sort=True)
        .agg(
            episodes=("episode_id", "size"),
            wells=("well", "nunique"),
            rows=("rows", "sum"),
            episode_sse=("episode_sse", "sum"),
        )
        .reset_index()
    )
    result["episode_fraction"] = result["episodes"] / len(episodes)
    result["sse_fraction"] = result["episode_sse"] / total_sse
    return result.sort_values(
        ["episode_sse", "cause"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)


def add_global_fractions(
    frame: pd.DataFrame,
    *,
    total_episodes: int,
    total_sse: float,
) -> pd.DataFrame:
    result = frame.copy()
    result["global_episode_fraction"] = result["episodes"] / total_episodes
    result["global_sse_fraction"] = result["episode_sse"] / total_sse
    return result


def grouped_cause_readout(
    episodes: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    total_sse = float(episodes["episode_sse"].sum())
    grouped = (
        episodes.groupby(group_columns + ["cause"], observed=False, sort=True)
        .agg(
            episodes=("episode_id", "size"),
            wells=("well", "nunique"),
            rows=("rows", "sum"),
            episode_sse=("episode_sse", "sum"),
        )
        .reset_index()
    )
    return add_global_fractions(
        grouped,
        total_episodes=len(episodes),
        total_sse=total_sse,
    )


def mechanism_overlap_readouts(
    episodes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize preregistered mechanism flags without exclusive priority."""

    flags = {
        "initial_condition_support_miss": episodes[
            "initial_condition_support_miss"
        ].to_numpy(bool),
        "transition": episodes["transition_overlap"].to_numpy(bool),
        "emission": episodes["emission_overlap"].to_numpy(bool),
        "resampling": episodes["resampling_overlap"].to_numpy(bool),
        "within_seed_multiplicity": (
            episodes["within_seed_multiplicity_row_fraction"].to_numpy(
                np.float64
            )
            >= 0.50
        ),
        "across_seed_multiplicity": (
            episodes["across_seed_aggregation_row_fraction"].to_numpy(
                np.float64
            )
            >= 0.50
        ),
        "particle_support_shortage": (
            episodes["support_shortage_row_fraction"].to_numpy(np.float64)
            >= 0.50
        ),
        "hard_clamp_shortage": (
            episodes["clamp_outside_row_fraction"].to_numpy(np.float64)
            >= 0.50
        ),
    }
    total_sse = float(episodes["episode_sse"].sum())
    flag_rows: list[dict[str, Any]] = []
    for name, mask in flags.items():
        flag_rows.append(
            {
                "mechanism_flag": name,
                "episodes": int(mask.sum()),
                "wells": int(episodes.loc[mask, "well"].nunique()),
                "rows": int(episodes.loc[mask, "rows"].sum()),
                "episode_sse": float(episodes.loc[mask, "episode_sse"].sum()),
                "episode_fraction": float(mask.mean()),
                "sse_fraction": float(
                    episodes.loc[mask, "episode_sse"].sum()
                    / max(total_sse, 1.0e-12)
                ),
            }
        )

    pair_rows: list[dict[str, Any]] = []
    names = list(flags)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            mask = flags[left] & flags[right]
            pair_rows.append(
                {
                    "left_flag": left,
                    "right_flag": right,
                    "episodes": int(mask.sum()),
                    "wells": int(episodes.loc[mask, "well"].nunique()),
                    "rows": int(episodes.loc[mask, "rows"].sum()),
                    "episode_sse": float(
                        episodes.loc[mask, "episode_sse"].sum()
                    ),
                    "episode_fraction": float(mask.mean()),
                    "sse_fraction": float(
                        episodes.loc[mask, "episode_sse"].sum()
                        / max(total_sse, 1.0e-12)
                    ),
                }
            )

    patterns = np.empty(len(episodes), dtype=object)
    for row_index in range(len(episodes)):
        patterns[row_index] = (
            "+".join(
                name
                for name, values in flags.items()
                if bool(values[row_index])
            )
            or "none"
        )
    pattern_summary = (
        episodes.assign(mechanism_pattern=patterns)
        .groupby("mechanism_pattern", sort=True)
        .agg(
            episodes=("episode_id", "size"),
            wells=("well", "nunique"),
            rows=("rows", "sum"),
            episode_sse=("episode_sse", "sum"),
        )
        .reset_index()
    )
    pattern_summary["episode_fraction"] = (
        pattern_summary["episodes"] / len(episodes)
    )
    pattern_summary["sse_fraction"] = (
        pattern_summary["episode_sse"] / max(total_sse, 1.0e-12)
    )
    return (
        pd.DataFrame(flag_rows).sort_values(
            ["episode_sse", "mechanism_flag"],
            ascending=[False, True],
            kind="stable",
        ),
        pd.DataFrame(pair_rows).sort_values(
            ["episode_sse", "left_flag", "right_flag"],
            ascending=[False, True, True],
            kind="stable",
        ),
        pattern_summary.sort_values(
            ["episode_sse", "mechanism_pattern"],
            ascending=[False, True],
            kind="stable",
        ),
    )


def episode_diagnostic_by_cause(episodes: pd.DataFrame) -> pd.DataFrame:
    fields = (
        "raw_gr_missing_fraction",
        "resampled_seed_fraction_mean",
        "ess_mean",
        "unique_ancestor_fraction_mean",
        "max_offspring_fraction_mean",
        "predictive_truth_mass_r05_mean",
        "filtered_truth_mass_r05_mean",
        "postresample_truth_mass_r05_mean",
        "truth_close_seed_fraction_mean",
        "best_seed_abs_error_ft_mean",
        "seed_prediction_std_ft_mean",
        "aggregation_abs_penalty_vs_best_seed_ft_mean",
        "within_seed_multiplicity_row_fraction",
        "across_seed_aggregation_row_fraction",
        "support_shortage_row_fraction",
        "clamp_outside_row_fraction",
        "start_suffix_fraction",
        "episode_suffix_fraction",
        "rows_from_last_within5_to_episode_start",
        "rows_from_episode_end_to_next_within5",
        "pre128_error_slope_ft_per_row",
        "pre128_error_start_ft",
        "pre128_error_end_ft",
        "error_std_ft",
        "error_slope_ft_per_row",
        "max_abs_error_ft",
    )
    rows: list[dict[str, Any]] = []
    grouped: list[tuple[str, pd.DataFrame]] = [
        ("__all__", episodes),
        *[
            (str(cause), group)
            for cause, group in episodes.groupby(
                "cause", observed=False, sort=True
            )
        ],
    ]
    for cause, group in grouped:
        record: dict[str, Any] = {
            "cause": cause,
            "episodes": int(len(group)),
            "wells": int(group["well"].nunique()),
            "rows": int(group["rows"].sum()),
            "episode_sse": float(group["episode_sse"].sum()),
        }
        for field in fields:
            values = pd.to_numeric(group[field], errors="coerce").to_numpy(
                np.float64
            )
            finite = values[np.isfinite(values)]
            record[f"{field}_mean"] = (
                float(np.mean(finite)) if len(finite) else np.nan
            )
            record[f"{field}_median"] = (
                float(np.median(finite)) if len(finite) else np.nan
            )
        rows.append(record)
    return pd.DataFrame(rows)


def canonical_group_fold_map(source_rows: Path) -> dict[str, int]:
    """Reproduce exp072/sklearn GroupKFold's size-balanced well assignment."""

    if sha256_path(source_rows) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("canonical source-row SHA changed")
    counts: dict[str, int] = {}
    rows = 0
    for chunk in pd.read_csv(
        source_rows,
        usecols=["well"],
        dtype={"well": str},
        chunksize=500_000,
    ):
        rows += len(chunk)
        for well, count in chunk["well"].value_counts(sort=False).items():
            counts[str(well)] = counts.get(str(well), 0) + int(count)
    if rows != EXPECTED_SOURCE_ROWS or len(counts) != EXPECTED_SOURCE_WELLS:
        raise RuntimeError(
            f"canonical fold source coverage changed: {rows}/{len(counts)}"
        )
    well_names = np.asarray(sorted(counts), dtype=object)
    group_counts = np.asarray(
        [counts[str(well)] for well in well_names], dtype=np.int64
    )
    # This is the exact deterministic assignment used by sklearn GroupKFold
    # without shuffle and by exp263's canonical cache builder.
    size_order = np.argsort(group_counts, kind="stable")[::-1]
    fold_sizes = np.zeros(5, dtype=np.int64)
    fold_by_sorted_group = np.empty(len(well_names), dtype=np.uint8)
    for group_position in size_order:
        fold = int(np.argmin(fold_sizes))
        fold_sizes[fold] += int(group_counts[group_position])
        fold_by_sorted_group[group_position] = fold
    return {
        str(well): int(fold_by_sorted_group[index])
        for index, well in enumerate(well_names)
    }


def assign_episode_rows(
    ledger: pd.DataFrame,
    episode_asset: pd.DataFrame,
) -> pd.DataFrame:
    by_well = {
        str(well): group.sort_values("row_idx", kind="stable")
        for well, group in ledger.groupby("well", sort=False)
    }
    parts: list[pd.DataFrame] = []
    for episode in episode_asset.itertuples(index=False):
        rows = by_well[str(episode.well)]
        selected = rows.loc[
            (rows["row_idx"] >= int(episode.start_row_idx))
            & (rows["row_idx"] < int(episode.end_row_idx_exclusive))
        ].copy()
        if len(selected) != int(episode.rows):
            raise RuntimeError(
                f"{episode.episode_id}: row coverage {len(selected)}/{episode.rows}"
            )
        selected.insert(0, "episode_id", str(episode.episode_id))
        parts.append(selected)
    result = pd.concat(parts, ignore_index=True)
    if len(result) != EXPECTED_EPISODE_ROWS:
        raise RuntimeError(f"episode row ledger count changed: {len(result)}")
    return result


def recapture_readout(
    ledger: pd.DataFrame,
    episodes: pd.DataFrame,
    *,
    mass_floor: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Locate first particle/seed basin loss and any later within-audit return."""

    by_well = {
        str(well): group.sort_values("row_idx", kind="stable")
        for well, group in ledger.groupby("well", sort=False)
    }
    signal_columns = {
        "predictive_particle_mass": "predictive_truth_mass_r05",
        "filtered_particle_mass": "filtered_truth_mass_r05",
        "postresample_particle_mass": "postresample_truth_mass_r05",
        "truth_close_seed_fraction": "truth_close_seed_fraction",
    }
    rows: list[dict[str, Any]] = []
    for episode in episodes.itertuples(index=False):
        scope = by_well[str(episode.well)]
        scope = scope.loc[
            (scope["row_idx"] >= int(episode.audit_start_row_idx))
            & (scope["row_idx"] < int(episode.end_row_idx_exclusive))
        ].sort_values("row_idx", kind="stable")
        record: dict[str, Any] = {
            "episode_id": str(episode.episode_id),
            "well": str(episode.well),
            "cause": str(episode.cause),
            "episode_rows": int(episode.rows),
            "episode_sse": float(episode.episode_sse),
            "episode_start_row_idx": int(episode.start_row_idx),
            "episode_end_row_idx_exclusive": int(
                episode.end_row_idx_exclusive
            ),
            "rows_from_last_within5_to_episode_start": float(
                episode.rows_from_last_within5_to_episode_start
            ),
            "rows_from_episode_end_to_next_within5": float(
                episode.rows_from_episode_end_to_next_within5
            ),
        }
        for signal, column in signal_columns.items():
            values = scope[column].to_numpy(np.float64)
            indices = scope["row_idx"].to_numpy(np.int64)
            onset_mask = indices >= int(episode.start_row_idx)
            onset_values = values[onset_mask]
            onset_indices = indices[onset_mask]
            loss_positions = np.flatnonzero(values < mass_floor)
            first_loss_position = (
                int(loss_positions[0]) if loss_positions.size else None
            )
            recapture_position: int | None = None
            if first_loss_position is not None:
                later = np.flatnonzero(
                    values[first_loss_position + 1 :] >= mass_floor
                )
                if later.size:
                    recapture_position = (
                        first_loss_position + 1 + int(later[0])
                    )
            first_loss_row = (
                float(indices[first_loss_position])
                if first_loss_position is not None
                else np.nan
            )
            recapture_row = (
                float(indices[recapture_position])
                if recapture_position is not None
                else np.nan
            )
            record[f"{signal}_first_loss_row_idx"] = first_loss_row
            record[f"{signal}_first_loss_offset_from_episode_start"] = (
                first_loss_row - int(episode.start_row_idx)
                if np.isfinite(first_loss_row)
                else np.nan
            )
            record[f"{signal}_recapture_row_idx"] = recapture_row
            record[f"{signal}_recapture_offset_from_episode_start"] = (
                recapture_row - int(episode.start_row_idx)
                if np.isfinite(recapture_row)
                else np.nan
            )
            record[f"{signal}_recapture_latency_after_loss_rows"] = (
                recapture_row - first_loss_row
                if np.isfinite(first_loss_row) and np.isfinite(recapture_row)
                else np.nan
            )
            record[f"{signal}_lost"] = first_loss_position is not None
            record[f"{signal}_recaptured_within_audit"] = (
                recapture_position is not None
            )
            record[f"{signal}_ends_below_floor"] = bool(
                len(values) and values[-1] < mass_floor
            )

            onset_loss_positions = np.flatnonzero(onset_values < mass_floor)
            onset_first_loss_position = (
                int(onset_loss_positions[0])
                if onset_loss_positions.size
                else None
            )
            onset_recapture_position: int | None = None
            if onset_first_loss_position is not None:
                later = np.flatnonzero(
                    onset_values[onset_first_loss_position + 1 :]
                    >= mass_floor
                )
                if later.size:
                    onset_recapture_position = (
                        onset_first_loss_position + 1 + int(later[0])
                    )
            onset_loss_row = (
                float(onset_indices[onset_first_loss_position])
                if onset_first_loss_position is not None
                else np.nan
            )
            onset_recapture_row = (
                float(onset_indices[onset_recapture_position])
                if onset_recapture_position is not None
                else np.nan
            )
            record[f"{signal}_onset_first_loss_row_idx"] = onset_loss_row
            record[f"{signal}_onset_first_loss_offset_rows"] = (
                onset_loss_row - int(episode.start_row_idx)
                if np.isfinite(onset_loss_row)
                else np.nan
            )
            record[f"{signal}_onset_recapture_row_idx"] = onset_recapture_row
            record[f"{signal}_onset_recapture_offset_rows"] = (
                onset_recapture_row - int(episode.start_row_idx)
                if np.isfinite(onset_recapture_row)
                else np.nan
            )
            record[f"{signal}_onset_recapture_latency_rows"] = (
                onset_recapture_row - onset_loss_row
                if np.isfinite(onset_loss_row)
                and np.isfinite(onset_recapture_row)
                else np.nan
            )
            record[f"{signal}_onset_lost"] = (
                onset_first_loss_position is not None
            )
            record[f"{signal}_onset_recaptured"] = (
                onset_recapture_position is not None
            )
        rows.append(record)

    episode_readout = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    total_sse = float(episode_readout["episode_sse"].sum())
    for signal in signal_columns:
        lost = episode_readout[f"{signal}_onset_lost"].to_numpy(bool)
        recaptured = episode_readout[
            f"{signal}_onset_recaptured"
        ].to_numpy(bool)
        ends_below = episode_readout[
            f"{signal}_ends_below_floor"
        ].to_numpy(bool)
        latency = pd.to_numeric(
            episode_readout[
                f"{signal}_onset_recapture_latency_rows"
            ],
            errors="coerce",
        ).to_numpy(np.float64)
        finite_latency = latency[np.isfinite(latency)]
        no_recapture = lost & ~recaptured
        summary_rows.append(
            {
                "signal": signal,
                "episodes": int(len(episode_readout)),
                "lost_episodes": int(lost.sum()),
                "lost_episode_fraction": float(lost.mean()),
                "recaptured_within_audit_episodes": int(
                    (lost & recaptured).sum()
                ),
                "recaptured_given_loss_fraction": float(
                    (lost & recaptured).sum() / max(int(lost.sum()), 1)
                ),
                "no_recapture_episodes": int(no_recapture.sum()),
                "no_recapture_sse": float(
                    episode_readout.loc[no_recapture, "episode_sse"].sum()
                ),
                "no_recapture_sse_fraction": float(
                    episode_readout.loc[
                        no_recapture, "episode_sse"
                    ].sum()
                    / max(total_sse, 1.0e-12)
                ),
                "ends_below_floor_episodes": int(ends_below.sum()),
                "recapture_latency_p25_rows": float(
                    np.quantile(finite_latency, 0.25)
                )
                if len(finite_latency)
                else np.nan,
                "recapture_latency_p50_rows": float(
                    np.quantile(finite_latency, 0.50)
                )
                if len(finite_latency)
                else np.nan,
                "recapture_latency_p75_rows": float(
                    np.quantile(finite_latency, 0.75)
                )
                if len(finite_latency)
                else np.nan,
                "recapture_latency_p95_rows": float(
                    np.quantile(finite_latency, 0.95)
                )
                if len(finite_latency)
                else np.nan,
            }
        )
    output_latency = pd.to_numeric(
        episode_readout["rows_from_episode_end_to_next_within5"],
        errors="coerce",
    ).to_numpy(np.float64)
    output_recaptured = np.isfinite(output_latency)
    output_no_recapture = ~output_recaptured
    finite_output_latency = output_latency[output_recaptured]
    summary_rows.append(
        {
            "signal": "fixed_output_error_within5",
            "episodes": int(len(episode_readout)),
            "lost_episodes": int(len(episode_readout)),
            "lost_episode_fraction": 1.0,
            "recaptured_within_audit_episodes": int(output_recaptured.sum()),
            "recaptured_given_loss_fraction": float(
                output_recaptured.mean()
            ),
            "no_recapture_episodes": int(output_no_recapture.sum()),
            "no_recapture_sse": float(
                episode_readout.loc[
                    output_no_recapture, "episode_sse"
                ].sum()
            ),
            "no_recapture_sse_fraction": float(
                episode_readout.loc[
                    output_no_recapture, "episode_sse"
                ].sum()
                / max(total_sse, 1.0e-12)
            ),
            "ends_below_floor_episodes": int(output_no_recapture.sum()),
            "recapture_latency_p25_rows": float(
                np.quantile(finite_output_latency, 0.25)
            )
            if len(finite_output_latency)
            else np.nan,
            "recapture_latency_p50_rows": float(
                np.quantile(finite_output_latency, 0.50)
            )
            if len(finite_output_latency)
            else np.nan,
            "recapture_latency_p75_rows": float(
                np.quantile(finite_output_latency, 0.75)
            )
            if len(finite_output_latency)
            else np.nan,
            "recapture_latency_p95_rows": float(
                np.quantile(finite_output_latency, 0.95)
            )
            if len(finite_output_latency)
            else np.nan,
        }
    )
    return episode_readout, pd.DataFrame(summary_rows)


def condition_summary(
    episode_rows: pd.DataFrame,
    conditions: dict[str, np.ndarray],
) -> pd.DataFrame:
    squared_error = episode_rows["error_ft"].to_numpy(np.float64) ** 2
    total_sse = float(squared_error.sum())
    rows: list[dict[str, Any]] = []
    for name, condition in conditions.items():
        mask = np.asarray(condition, dtype=bool)
        rows.append(
            {
                "condition": name,
                "rows": int(mask.sum()),
                "row_fraction": float(mask.mean()),
                "wells": int(episode_rows.loc[mask, "well"].nunique()),
                "episodes": int(episode_rows.loc[mask, "episode_id"].nunique()),
                "sse": float(squared_error[mask].sum()),
                "sse_fraction": float(squared_error[mask].sum())
                / max(total_sse, 1.0e-12),
                "rmse_ft": float(np.sqrt(np.mean(squared_error[mask])))
                if mask.any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["sse", "condition"], ascending=[False, True], kind="stable"
    )


def aggregation_alternative_summary(
    episode_rows: pd.DataFrame,
    *,
    group_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Compare target-free PF readouts and a clearly labelled oracle bound.

    All non-oracle predictions are already present in the unchanged replay
    ledger.  This function only evaluates them on the frozen PF-offset rows.
    """

    prediction_columns = {
        "fixed_arithmetic_seed_mean": "fixed_likpf_mean",
        "target_free_seed_median": "median_seed_prediction",
        "predictive_mean_before_gr": "predictive_mean_tvt",
        "filtered_mean_before_resampling": "filtered_mean_tvt",
        "postresample_mean": "postresample_mean_tvt",
    }
    groupby_key: str | list[str]
    if len(group_columns) == 1:
        groupby_key = group_columns[0]
    else:
        groupby_key = list(group_columns)
    grouped: Iterable[tuple[Any, pd.DataFrame]]
    if group_columns:
        grouped = episode_rows.groupby(
            groupby_key, observed=False, sort=True, dropna=False
        )
    else:
        grouped = [((), episode_rows)]

    rows: list[dict[str, Any]] = []
    for key, group in grouped:
        key_values = key if isinstance(key, tuple) else (key,)
        labels = dict(zip(group_columns, key_values))
        truth = group["true_tvt"].to_numpy(np.float64)
        control_error = (
            group["fixed_likpf_mean"].to_numpy(np.float64) - truth
        )
        control_squared_error = control_error * control_error
        control_sse = float(control_squared_error.sum())
        for readout, column in prediction_columns.items():
            error = group[column].to_numpy(np.float64) - truth
            squared_error = error * error
            rows.append(
                {
                    **labels,
                    "readout": readout,
                    "target_free": True,
                    "oracle": False,
                    "rows": int(len(group)),
                    "wells": int(group["well"].nunique()),
                    "episodes": int(group["episode_id"].nunique()),
                    "sse": float(squared_error.sum()),
                    "rmse_ft": float(np.sqrt(np.mean(squared_error))),
                    "sse_delta_vs_fixed_mean": float(
                        squared_error.sum() - control_sse
                    ),
                    "sse_relative_vs_fixed_mean": float(
                        squared_error.sum() / max(control_sse, 1.0e-12)
                    ),
                    "row_abs_error_improved_fraction": float(
                        np.mean(np.abs(error) < np.abs(control_error))
                    ),
                    "row_abs_error_tied_fraction": float(
                        np.mean(np.abs(error) == np.abs(control_error))
                    ),
                    "mean_signed_error_ft": float(np.mean(error)),
                }
            )

        # The closest seed uses truth and is only an unattainable upper bound.
        best_abs_error = group["best_seed_abs_error_ft"].to_numpy(np.float64)
        best_squared_error = best_abs_error * best_abs_error
        rows.append(
            {
                **labels,
                "readout": "oracle_best_seed_per_row",
                "target_free": False,
                "oracle": True,
                "rows": int(len(group)),
                "wells": int(group["well"].nunique()),
                "episodes": int(group["episode_id"].nunique()),
                "sse": float(best_squared_error.sum()),
                "rmse_ft": float(np.sqrt(np.mean(best_squared_error))),
                "sse_delta_vs_fixed_mean": float(
                    best_squared_error.sum() - control_sse
                ),
                "sse_relative_vs_fixed_mean": float(
                    best_squared_error.sum() / max(control_sse, 1.0e-12)
                ),
                "row_abs_error_improved_fraction": float(
                    np.mean(best_abs_error < np.abs(control_error))
                ),
                "row_abs_error_tied_fraction": float(
                    np.mean(best_abs_error == np.abs(control_error))
                ),
                "mean_signed_error_ft": np.nan,
            }
        )
    return pd.DataFrame(rows)


def stage_effect_summary(
    episode_rows: pd.DataFrame,
    *,
    group_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Measure the immediate positional and SSE effect of each PF stage."""

    stage_pairs = (
        (
            "gr_update",
            "predictive_mean_tvt",
            "filtered_mean_tvt",
        ),
        (
            "conditional_resampling",
            "filtered_mean_tvt",
            "postresample_mean_tvt",
        ),
    )
    groupby_key: str | list[str]
    if len(group_columns) == 1:
        groupby_key = group_columns[0]
    else:
        groupby_key = list(group_columns)
    grouped: Iterable[tuple[Any, pd.DataFrame]]
    if group_columns:
        grouped = episode_rows.groupby(
            groupby_key, observed=False, sort=True, dropna=False
        )
    else:
        grouped = [((), episode_rows)]

    rows: list[dict[str, Any]] = []
    for key, group in grouped:
        key_values = key if isinstance(key, tuple) else (key,)
        labels = dict(zip(group_columns, key_values))
        truth = group["true_tvt"].to_numpy(np.float64)
        for stage, before_column, after_column in stage_pairs:
            before_error = group[before_column].to_numpy(np.float64) - truth
            after_error = group[after_column].to_numpy(np.float64) - truth
            before_sse = float(np.sum(before_error * before_error))
            after_sse = float(np.sum(after_error * after_error))
            displacement = after_error - before_error
            rows.append(
                {
                    **labels,
                    "stage": stage,
                    "rows": int(len(group)),
                    "wells": int(group["well"].nunique()),
                    "episodes": int(group["episode_id"].nunique()),
                    "before_sse": before_sse,
                    "after_sse": after_sse,
                    "sse_delta_after_minus_before": after_sse - before_sse,
                    "after_to_before_sse_ratio": after_sse
                    / max(before_sse, 1.0e-12),
                    "toward_truth_row_fraction": float(
                        np.mean(np.abs(after_error) < np.abs(before_error))
                    ),
                    "away_from_truth_row_fraction": float(
                        np.mean(np.abs(after_error) > np.abs(before_error))
                    ),
                    "mean_signed_displacement_ft": float(
                        np.mean(displacement)
                    ),
                    "mean_abs_displacement_ft": float(
                        np.mean(np.abs(displacement))
                    ),
                    "p95_abs_displacement_ft": float(
                        np.quantile(np.abs(displacement), 0.95)
                    ),
                    "same_direction_as_final_error_fraction": float(
                        np.mean(
                            displacement
                            * (
                                group["error_ft"].to_numpy(np.float64)
                            )
                            > 0.0
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def continuous_summary(episode_rows: pd.DataFrame) -> pd.DataFrame:
    fields = (
        "error_ft",
        "predictive_truth_mass_r05",
        "filtered_truth_mass_r05",
        "postresample_truth_mass_r05",
        "predictive_candidate_mass_r05",
        "filtered_candidate_mass_r05",
        "postresample_candidate_mass_r05",
        "transition_truth_vs_candidate_log_ratio_delta",
        "emission_truth_vs_candidate_log_ratio_delta",
        "resampling_truth_vs_candidate_log_ratio_delta",
        "ess_mean",
        "resampled_seed_fraction",
        "unique_ancestor_fraction",
        "max_offspring_fraction",
        "transition_escape_seed_fraction",
        "emission_escape_seed_fraction",
        "resampling_extinction_seed_fraction",
        "within_seed_multiplicity_fraction",
        "truth_close_seed_fraction",
        "candidate_close_seed_fraction",
        "best_seed_abs_error_ft",
        "seed_prediction_std_ft",
        "aggregation_abs_penalty_vs_best_seed_ft",
        "predictive_mean_error_ft",
        "filtered_mean_error_ft",
        "postresample_mean_error_ft",
        "predictive_rate_error",
        "filtered_rate_error",
        "postresample_rate_error",
    )
    rows: list[dict[str, Any]] = []
    for field in fields:
        values = pd.to_numeric(episode_rows[field], errors="coerce").to_numpy(
            np.float64
        )
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "field": field,
                "rows": int(len(finite)),
                "mean": float(np.mean(finite)) if len(finite) else np.nan,
                "std": float(np.std(finite)) if len(finite) else np.nan,
                "p05": float(np.quantile(finite, 0.05)) if len(finite) else np.nan,
                "p25": float(np.quantile(finite, 0.25)) if len(finite) else np.nan,
                "p50": float(np.quantile(finite, 0.50)) if len(finite) else np.nan,
                "p75": float(np.quantile(finite, 0.75)) if len(finite) else np.nan,
                "p95": float(np.quantile(finite, 0.95)) if len(finite) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def correlation_summary(episode_rows: pd.DataFrame) -> pd.DataFrame:
    target_columns = (
        "error_ft",
        "predictive_mean_error_ft",
        "filtered_mean_error_ft",
        "postresample_mean_error_ft",
        "predictive_rate_error",
        "filtered_rate_error",
        "postresample_rate_error",
        "ess_mean",
        "resampled_seed_fraction",
        "unique_ancestor_fraction",
        "max_offspring_fraction",
        "transition_escape_seed_fraction",
        "emission_escape_seed_fraction",
        "resampling_extinction_seed_fraction",
        "within_seed_multiplicity_fraction",
        "truth_close_seed_fraction",
        "best_seed_abs_error_ft",
        "seed_prediction_std_ft",
    )
    numeric = episode_rows[list(target_columns)].apply(
        pd.to_numeric, errors="coerce"
    )
    spearman = numeric.corr(method="spearman")
    pearson = numeric.corr(method="pearson")
    rows: list[dict[str, Any]] = []
    for left_index, left in enumerate(target_columns):
        for right in target_columns[left_index + 1 :]:
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "spearman": float(spearman.loc[left, right]),
                    "pearson": float(pearson.loc[left, right]),
                    "abs_spearman": abs(float(spearman.loc[left, right])),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["abs_spearman", "left", "right"],
        ascending=[False, True, True],
        kind="stable",
    )


def timing_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stage, column in (
        ("transition", "first_transition_effect_row_idx"),
        ("emission", "first_emission_effect_row_idx"),
        ("resampling", "first_resampling_effect_row_idx"),
    ):
        offset = (
            pd.to_numeric(episodes[column], errors="coerce")
            - episodes["start_row_idx"].to_numpy(np.float64)
        )
        finite = offset[np.isfinite(offset)].to_numpy(np.float64)
        rows.append(
            {
                "stage": stage,
                "episodes": int(len(finite)),
                "pre_onset_episodes": int(np.sum(finite < 0.0)),
                "at_or_after_onset_episodes": int(np.sum(finite >= 0.0)),
                "p05_offset_rows": float(np.quantile(finite, 0.05))
                if len(finite)
                else np.nan,
                "p25_offset_rows": float(np.quantile(finite, 0.25))
                if len(finite)
                else np.nan,
                "p50_offset_rows": float(np.quantile(finite, 0.50))
                if len(finite)
                else np.nan,
                "p75_offset_rows": float(np.quantile(finite, 0.75))
                if len(finite)
                else np.nan,
                "p95_offset_rows": float(np.quantile(finite, 0.95))
                if len(finite)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def merge_threshold_sensitivity(frames: list[pd.DataFrame]) -> pd.DataFrame:
    keys = [
        "basin_radius_ft",
        "mass_floor",
        "log_odds_effect",
        "dominant_row_fraction",
        "stage",
    ]
    merged = (
        pd.concat(frames, ignore_index=True)
        .groupby(keys, observed=False, sort=True)
        .agg(episodes=("episodes", "sum"), episode_sse=("episode_sse", "sum"))
        .reset_index()
    )
    merged["episode_fraction"] = merged["episodes"] / EXPECTED_EPISODES
    merged["sse_fraction"] = merged["episode_sse"] / EXPECTED_EPISODE_SSE
    return merged


def summarize_threshold_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stage, group in frame.groupby("stage", observed=False, sort=True):
        fractions = group["sse_fraction"].to_numpy(np.float64)
        primary = group.loc[
            np.isclose(group["basin_radius_ft"], 5.0)
            & np.isclose(group["mass_floor"], 0.01)
            & np.isclose(group["log_odds_effect"], np.log(3.0))
            & np.isclose(group["dominant_row_fraction"], 0.50)
        ]
        if len(primary) != 1:
            raise RuntimeError(f"{stage}: primary sensitivity row missing")
        rows.append(
            {
                "stage": stage,
                "parameter_combinations": int(len(group)),
                "nonzero_combinations": int(np.sum(fractions > 0.0)),
                "nonzero_combination_fraction": float(
                    np.mean(fractions > 0.0)
                ),
                "sse_fraction_min": float(np.min(fractions)),
                "sse_fraction_p25": float(np.quantile(fractions, 0.25)),
                "sse_fraction_median": float(np.median(fractions)),
                "sse_fraction_p75": float(np.quantile(fractions, 0.75)),
                "sse_fraction_max": float(np.max(fractions)),
                "primary_sse_fraction": float(
                    primary["sse_fraction"].iloc[0]
                ),
                "primary_episodes": int(primary["episodes"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if len(args.shard) != 4:
        raise ValueError("exactly four shard roots are required")
    episode_asset = pd.read_csv(
        args.episodes, dtype={"episode_id": str, "well": str}
    )
    well_asset = pd.read_csv(args.target_wells, dtype={"well": str})
    fold_map = canonical_group_fold_map(args.source_rows)
    shard_manifests: list[dict[str, Any]] = []
    ledgers: list[pd.DataFrame] = []
    episodes: list[pd.DataFrame] = []
    wells: list[pd.DataFrame] = []
    sensitivities: list[pd.DataFrame] = []
    for shard_root in args.shard:
        summary_path = require_artifact(shard_root, f"{PREFIX}_summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        shard_index = int(summary["active_shard"])
        if summary["run_stage"] != "full":
            raise RuntimeError(f"{shard_root}: not a full run")
        row_path = require_artifact(shard_root, f"{PREFIX}_row_ledger.csv.gz")
        episode_path = require_artifact(
            shard_root, f"{PREFIX}_episode_summary.csv"
        )
        well_path = require_artifact(shard_root, f"{PREFIX}_well_manifest.csv")
        sensitivity_path = require_artifact(
            shard_root, f"{PREFIX}_threshold_sensitivity.csv"
        )
        ledgers.append(
            pd.read_csv(row_path, dtype={"well": str, "id": str})
        )
        episodes.append(
            pd.read_csv(
                episode_path, dtype={"episode_id": str, "well": str}
            )
        )
        wells.append(pd.read_csv(well_path, dtype={"well": str}))
        sensitivities.append(pd.read_csv(sensitivity_path))
        shard_manifests.append(
            {
                "shard_index": shard_index,
                "root": str(shard_root),
                "summary": summary,
                "summary_sha256": sha256_path(summary_path),
                "row_ledger_sha256": sha256_path(row_path),
                "row_ledger_decompressed_sha256": sha256_path(
                    row_path, decompressed=True
                ),
                "episode_summary_sha256": sha256_path(episode_path),
                "well_manifest_sha256": sha256_path(well_path),
                "threshold_sensitivity_sha256": sha256_path(sensitivity_path),
            }
        )
    if sorted(item["shard_index"] for item in shard_manifests) != [0, 1, 2, 3]:
        raise RuntimeError("full shard indices are incomplete or duplicated")

    ledger = pd.concat(ledgers, ignore_index=True).sort_values(
        ["well", "row_idx"], kind="stable"
    )
    episode_frame = pd.concat(episodes, ignore_index=True).sort_values(
        ["well", "start_row_idx"], kind="stable"
    )
    well_frame = pd.concat(wells, ignore_index=True).sort_values(
        "well", kind="stable"
    )
    if well_frame["well"].duplicated().any():
        raise RuntimeError("duplicate wells across shards")
    if episode_frame["episode_id"].duplicated().any():
        raise RuntimeError("duplicate episodes across shards")
    if ledger[["well", "row_idx"]].duplicated().any():
        raise RuntimeError("duplicate audit rows across shards")
    actual = {
        "wells": len(well_frame),
        "suffix_rows": int(well_frame["suffix_rows"].sum()),
        "episodes": len(episode_frame),
        "episode_rows": int(episode_frame["rows"].sum()),
    }
    expected = {
        "wells": EXPECTED_WELLS,
        "suffix_rows": EXPECTED_SUFFIX_ROWS,
        "episodes": EXPECTED_EPISODES,
        "episode_rows": EXPECTED_EPISODE_ROWS,
    }
    if actual != expected:
        raise RuntimeError(f"strict merged coverage failed: {actual} != {expected}")
    if set(well_frame["well"]) != set(well_asset["well"]):
        raise RuntimeError("merged well IDs do not equal the fixed target asset")
    if set(episode_frame["episode_id"]) != set(episode_asset["episode_id"]):
        raise RuntimeError("merged episode IDs do not equal the fixed asset")
    if float(well_frame["parity_max_abs_ft"].max()) != 0.0:
        raise RuntimeError("a full shard has nonzero persisted prediction parity")
    if not np.isclose(
        float(episode_frame["episode_sse"].sum()),
        EXPECTED_EPISODE_SSE,
        rtol=1.0e-12,
        atol=1.0e-6,
    ):
        raise RuntimeError("merged episode SSE changed")

    asset_columns = [
        "episode_id",
        "audit_start_row_idx",
        "start_suffix_offset",
        "episode_start_md_since",
        "episode_end_md_since",
        "rows_from_last_within5_to_episode_start",
        "rows_from_episode_end_to_next_within5",
        "start_suffix_fraction",
        "episode_suffix_fraction",
        "pre128_error_slope_ft_per_row",
        "pre128_error_start_ft",
        "pre128_error_end_ft",
        "error_std_ft",
        "error_slope_ft_per_row",
        "max_abs_error_ft",
        "last_known_tvt",
        "last_known_md",
    ]
    episode_frame = episode_frame.merge(
        episode_asset[asset_columns],
        on="episode_id",
        how="left",
        validate="one_to_one",
    )
    episode_frame["error_sign"] = np.where(
        episode_frame["mean_error_ft"] >= 0.0, "positive", "negative"
    )
    episode_frame["canonical_outer_fold"] = episode_frame["well"].map(fold_map)
    if episode_frame["canonical_outer_fold"].isna().any():
        raise RuntimeError("a PF episode well is absent from the canonical fold map")
    episode_frame["canonical_outer_fold"] = episode_frame[
        "canonical_outer_fold"
    ].astype("int8")
    episode_frame["length_bucket"] = pd.cut(
        episode_frame["rows"],
        bins=[0, 255, 511, 1023, 2047, np.inf],
        labels=["128_255", "256_511", "512_1023", "1024_2047", "2048_plus"],
        right=True,
    )
    episode_frame["tail_start_bucket"] = pd.cut(
        episode_frame["episode_start_md_since"],
        bins=[-np.inf, 512, 1024, 2048, 4096, np.inf],
        labels=["le512", "512_1024", "1024_2048", "2048_4096", "4096_plus"],
        right=True,
    )
    slope = episode_frame["error_slope_ft_per_row"].to_numpy(np.float64)
    signed_growth = slope * np.sign(
        episode_frame["mean_error_ft"].to_numpy(np.float64)
    )
    episode_frame["offset_dynamics"] = np.select(
        [signed_growth > 0.001, signed_growth < -0.001],
        ["growing_magnitude", "shrinking_magnitude"],
        default="approximately_flat",
    )
    episode_frame["onset_suffix_bucket"] = pd.cut(
        episode_frame["start_suffix_fraction"],
        bins=[0.0, 0.25, 0.50, 0.75, 1.0],
        labels=["q1", "q2", "q3", "q4"],
        include_lowest=True,
        right=True,
    )
    overlap_columns = (
        "transition_overlap",
        "emission_overlap",
        "resampling_overlap",
        "initial_condition_support_miss",
    )
    episode_frame["overlap_pattern"] = episode_frame.apply(
        lambda row: "+".join(
            name.replace("_overlap", "")
            for name in overlap_columns
            if bool(row[name])
        )
        or "none",
        axis=1,
    )

    recapture_episodes, recapture_summary = recapture_readout(
        ledger,
        episode_frame,
        mass_floor=0.01,
    )
    audit_ledger_rows = len(ledger)
    episode_rows = assign_episode_rows(ledger, episode_asset)
    if not args.write_large_ledgers:
        # The downloaded shard ledgers remain the canonical row-level evidence.
        # Avoid retaining two near-identical wide merged frames during the
        # cross-angle summaries.
        del ledger
        gc.collect()
    error = episode_rows["error_ft"].to_numpy(np.float64)
    conditions = {
        "raw_gr_missing": episode_rows["raw_gr_missing"].to_numpy(bool),
        "raw_gr_observed": ~episode_rows["raw_gr_missing"].to_numpy(bool),
        "resampled_any_seed": (
            episode_rows["resampled_seed_fraction"].to_numpy(np.float64) > 0.0
        ),
        "resampled_majority_seeds": (
            episode_rows["resampled_seed_fraction"].to_numpy(np.float64) >= 0.5
        ),
        "low_ess_below_half_particles": (
            episode_rows["ess_mean"].to_numpy(np.float64) < 250.0
        ),
        "severe_ancestor_concentration": (
            episode_rows["unique_ancestor_fraction"].to_numpy(np.float64) < 0.10
        )
        | (
            episode_rows["max_offspring_fraction"].to_numpy(np.float64) >= 0.25
        ),
        "transition_escape_majority_seeds": (
            episode_rows["transition_escape_seed_fraction"].to_numpy(np.float64)
            >= 0.5
        ),
        "emission_escape_majority_seeds": (
            episode_rows["emission_escape_seed_fraction"].to_numpy(np.float64)
            >= 0.5
        ),
        "resampling_extinction_majority_seeds": (
            episode_rows[
                "resampling_extinction_seed_fraction"
            ].to_numpy(np.float64)
            >= 0.5
        ),
        "within_seed_multiplicity_majority": (
            episode_rows["within_seed_multiplicity_fraction"].to_numpy(np.float64)
            >= 0.5
        ),
        "predictive_truth_and_candidate_basin_coexist_1pct": (
            episode_rows["predictive_truth_mass_r05"].to_numpy(np.float64)
            >= 0.01
        )
        & (
            episode_rows["predictive_candidate_mass_r05"].to_numpy(np.float64)
            >= 0.01
        ),
        "filtered_truth_and_candidate_basin_coexist_1pct": (
            episode_rows["filtered_truth_mass_r05"].to_numpy(np.float64)
            >= 0.01
        )
        & (
            episode_rows["filtered_candidate_mass_r05"].to_numpy(np.float64)
            >= 0.01
        ),
        "postresample_truth_and_candidate_basin_coexist_1pct": (
            episode_rows["postresample_truth_mass_r05"].to_numpy(np.float64)
            >= 0.01
        )
        & (
            episode_rows[
                "postresample_candidate_mass_r05"
            ].to_numpy(np.float64)
            >= 0.01
        ),
        "postresample_truth_and_candidate_basin_coexist_5pct": (
            episode_rows["postresample_truth_mass_r05"].to_numpy(np.float64)
            >= 0.05
        )
        & (
            episode_rows[
                "postresample_candidate_mass_r05"
            ].to_numpy(np.float64)
            >= 0.05
        ),
        "postresample_candidate_odds_at_least3x_truth": (
            episode_rows["postresample_candidate_mass_r05"].to_numpy(
                np.float64
            )
            >= 3.0
            * episode_rows["postresample_truth_mass_r05"].to_numpy(
                np.float64
            )
        ),
        "truth_close_seed_exists_1pct": (
            episode_rows["truth_close_seed_fraction"].to_numpy(np.float64)
            >= 0.01
        ),
        "truth_and_candidate_close_seed_populations_exist_1pct": (
            episode_rows["truth_close_seed_fraction"].to_numpy(np.float64)
            >= 0.01
        )
        & (
            episode_rows["candidate_close_seed_fraction"].to_numpy(np.float64)
            >= 0.01
        ),
        "best_seed_within5": (
            episode_rows["best_seed_abs_error_ft"].to_numpy(np.float64) <= 5.0
        ),
        "best_seed_improves_at_least5": (
            np.abs(error)
            - episode_rows["best_seed_abs_error_ft"].to_numpy(np.float64)
            >= 5.0
        ),
        "best_seed_improves_at_least10": (
            np.abs(error)
            - episode_rows["best_seed_abs_error_ft"].to_numpy(np.float64)
            >= 10.0
        ),
        "truth_outside_particle_support_majority": (
            episode_rows[
                "postresample_truth_support_fraction"
            ].to_numpy(np.float64)
            < 0.5
        ),
        "truth_outside_hard_clamp": episode_rows[
            "truth_outside_clamp"
        ].to_numpy(bool),
        "transition_rate_error_same_sign": (
            episode_rows["predictive_rate_error"].to_numpy(np.float64) * error
            > 0.0
        ),
        "predictive_position_error_same_sign": (
            episode_rows["predictive_mean_error_ft"].to_numpy(np.float64) * error
            > 0.0
        ),
        "emission_moves_toward_truth": (
            np.abs(
                episode_rows["filtered_mean_error_ft"].to_numpy(np.float64)
            )
            < np.abs(
                episode_rows["predictive_mean_error_ft"].to_numpy(np.float64)
            )
        ),
        "resampling_moves_toward_truth": (
            np.abs(
                episode_rows["postresample_mean_error_ft"].to_numpy(np.float64)
            )
            < np.abs(
                episode_rows["filtered_mean_error_ft"].to_numpy(np.float64)
            )
        ),
    }

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    cause = cause_summary(episode_frame)
    (
        mechanism_flags,
        mechanism_pairs,
        mechanism_patterns,
    ) = mechanism_overlap_readouts(episode_frame)
    threshold = merge_threshold_sensitivity(sensitivities)
    threshold_summary = summarize_threshold_sensitivity(threshold)
    outputs: dict[str, Path] = {}

    def save(name: str, frame: pd.DataFrame, *, gzip_output: bool = False) -> None:
        suffix = ".csv.gz" if gzip_output else ".csv"
        path = output / f"{PREFIX}_{name}{suffix}"
        frame.to_csv(
            path,
            index=False,
            compression="gzip" if gzip_output else None,
        )
        outputs[name] = path

    if args.write_large_ledgers:
        save("row_ledger", ledger, gzip_output=True)
        save("episode_row_ledger", episode_rows, gzip_output=True)
    save("episode_summary", episode_frame)
    save("well_manifest", well_frame)
    save("cause_summary", cause)
    save("mechanism_flag_summary", mechanism_flags)
    save("mechanism_pair_overlap_summary", mechanism_pairs)
    save("mechanism_pattern_summary", mechanism_patterns)
    save("episode_diagnostic_by_cause", episode_diagnostic_by_cause(episode_frame))
    save("threshold_sensitivity", threshold)
    save("threshold_sensitivity_summary", threshold_summary)
    save("cause_by_error_sign", grouped_cause_readout(episode_frame, ["error_sign"]))
    save(
        "cause_by_canonical_outer_fold",
        grouped_cause_readout(episode_frame, ["canonical_outer_fold"]),
    )
    save(
        "cause_by_length_bucket",
        grouped_cause_readout(episode_frame, ["length_bucket"]),
    )
    save(
        "cause_by_tail_start_bucket",
        grouped_cause_readout(episode_frame, ["tail_start_bucket"]),
    )
    save(
        "cause_by_offset_dynamics",
        grouped_cause_readout(episode_frame, ["offset_dynamics"]),
    )
    save(
        "cause_by_onset_suffix_bucket",
        grouped_cause_readout(episode_frame, ["onset_suffix_bucket"]),
    )
    save(
        "overlap_summary",
        grouped_cause_readout(episode_frame, ["overlap_pattern"]),
    )
    save("row_condition_summary", condition_summary(episode_rows, conditions))
    save(
        "aggregation_alternative_summary",
        aggregation_alternative_summary(episode_rows),
    )
    episode_to_cause = episode_frame.set_index("episode_id")["cause"]
    episode_to_sign = episode_frame.set_index("episode_id")["error_sign"]
    episode_rows["cause"] = episode_rows["episode_id"].map(episode_to_cause)
    episode_rows["error_sign"] = episode_rows["episode_id"].map(episode_to_sign)
    if episode_rows[["cause", "error_sign"]].isna().any().any():
        raise RuntimeError("episode cause/sign mapping failed")
    save(
        "aggregation_alternative_by_cause",
        aggregation_alternative_summary(
            episode_rows,
            group_columns=("cause",),
        ),
    )
    save(
        "aggregation_alternative_by_error_sign",
        aggregation_alternative_summary(
            episode_rows,
            group_columns=("error_sign",),
        ),
    )
    save("stage_effect_summary", stage_effect_summary(episode_rows))
    save(
        "stage_effect_by_cause",
        stage_effect_summary(
            episode_rows,
            group_columns=("cause",),
        ),
    )
    episode_rows["gr_observation"] = np.where(
        episode_rows["raw_gr_missing"].to_numpy(bool),
        "imputed",
        "observed",
    )
    save(
        "stage_effect_by_raw_gr_missing",
        stage_effect_summary(
            episode_rows,
            group_columns=("gr_observation",),
        ),
    )
    save("row_continuous_summary", continuous_summary(episode_rows))
    save("row_correlation_summary", correlation_summary(episode_rows))
    save("timing_summary", timing_summary(episode_frame))
    save("recapture_episode_summary", recapture_episodes)
    save("recapture_stage_summary", recapture_summary)

    summary = {
        "experiment": EXPERIMENT,
        "status": "complete_full_four_shard_strict_merge",
        "counts": {
            **actual,
            "audit_ledger_rows": audit_ledger_rows,
            "episode_row_ledger_rows": len(episode_rows),
        },
        "episode_sse": float(episode_frame["episode_sse"].sum()),
        "parity_max_abs_ft": float(well_frame["parity_max_abs_ft"].max()),
        "cause_sse_fraction": {
            str(row.cause): float(row.sse_fraction)
            for row in cause.itertuples(index=False)
        },
        "runtime": {
            "sum_well_seconds": float(well_frame["elapsed_seconds"].sum()),
            "max_peak_rss_gb": float(well_frame["peak_rss_gb_after"].max()),
            "shard_elapsed_seconds": {
                str(item["shard_index"]): float(
                    item["summary"]["runtime"]["elapsed_seconds"]
                )
                for item in shard_manifests
            },
        },
        "fixed_assets": {
            "episodes_sha256": sha256_path(args.episodes),
            "target_wells_sha256": sha256_path(args.target_wells),
            "canonical_fold_source_sha256": sha256_path(args.source_rows),
        },
        "implementation": {
            "full_package_code": str(args.full_package_code),
            "full_package_code_sha256": sha256_path(args.full_package_code),
            "full_package_config": str(args.full_package_config),
            "full_package_config_sha256": sha256_path(
                args.full_package_config
            ),
        },
        "shards": shard_manifests,
        "artifacts": {
            name: {
                "path": str(path),
                "sha256": sha256_path(path),
                "decompressed_sha256": (
                    sha256_path(path, decompressed=True)
                    if path.suffix == ".gz"
                    else None
                ),
            }
            for name, path in outputs.items()
        },
        "guards": {
            "strict_well_coverage": True,
            "strict_episode_coverage": True,
            "strict_episode_row_coverage": True,
            "all_persisted_prediction_parity_zero": True,
            "no_duplicate_well_episode_or_audit_row": True,
            "canonical_row_evidence_is_four_sha_guarded_shard_ledgers": (
                not args.write_large_ledgers
            ),
            "redundant_merged_large_ledgers_written": args.write_large_ledgers,
        },
    }
    summary_path = output / f"{PREFIX}_merged_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
