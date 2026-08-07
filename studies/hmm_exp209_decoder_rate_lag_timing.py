#!/usr/bin/env python3
"""Compare pre-offset rate lag across frozen exp270 decoders.

This truth-late readout compares posterior mean, row-wise marginal MAP, and
the legal global Viterbi path on the same persistent-offset onset rings.  It
does not rerun the HMM, generate predictions, or fit a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from hmm_exp209_offset_cause_readout import (
    DEFAULT_CANDIDATES,
    iter_well_frames,
    load_well_inputs,
)
from scipy.stats import spearmanr

DEFAULT_EPISODES = Path(
    "studies/hmm_exp209_rate_lag_timing_20260726/"
    "episode_rate_lag_metrics.csv"
)
DEFAULT_OUTPUT = Path(
    "studies/hmm_exp209_decoder_rate_lag_timing_20260726"
)
RINGS = (
    ("000_016", 0, 16),
    ("016_064", 16, 64),
    ("064_128", 64, 128),
    ("128_256", 128, 256),
    ("256_512", 256, 512),
)
DECODERS = {
    "posterior_mean": "posterior_mean",
    "marginal_map": "marginal_map",
    "global_viterbi": "topk_path_1",
}
DECODER_METRICS = (
    "decoded_rate_abs",
    "rate_error_signed",
    "rate_error_abs",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chunksize", type=int, default=300_000)
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    resolved = path if path.is_absolute() else root / path
    if not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def safe_spearman(left: pd.Series, right: pd.Series) -> float | None:
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 3:
        return None
    x = left.loc[finite].to_numpy(np.float64)
    y = right.loc[finite].to_numpy(np.float64)
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return None
    return float(spearmanr(x, y).statistic)


def fraction_within(mask: pd.Series, eligible: pd.Series) -> float | None:
    count = int(eligible.sum())
    if count == 0:
        return None
    return float((mask & eligible).sum() / count)


def weighted_fraction_within(
    mask: pd.Series,
    eligible: pd.Series,
    weights: pd.Series,
) -> float | None:
    denominator = float(weights.loc[eligible].sum())
    if denominator <= 0.0:
        return None
    return float(weights.loc[mask & eligible].sum() / denominator)


def add_interval(
    record: dict[str, Any],
    decoder: str,
    label: str,
    start: int,
    end: int,
    arrays: dict[str, np.ndarray],
) -> None:
    record[f"{decoder}_{label}_rows"] = int(end - start)
    for metric, values in arrays.items():
        part = values[start:end]
        finite = part[np.isfinite(part)]
        record[f"{decoder}_{label}_{metric}_mean"] = (
            float(finite.mean()) if len(finite) else None
        )
        record[f"{decoder}_{label}_{metric}_median"] = (
            float(np.median(finite)) if len(finite) else None
        )


def pooled_interval_mean(
    group: pd.DataFrame,
    decoder: str,
    label: str,
    metric: str,
) -> float | None:
    value = f"{decoder}_{label}_{metric}_mean"
    rows = f"{decoder}_{label}_rows"
    finite = group[value].notna() & (group[rows] > 0)
    denominator = float(group.loc[finite, rows].sum())
    if denominator <= 0.0:
        return None
    numerator = float(
        (group.loc[finite, value] * group.loc[finite, rows]).sum()
    )
    return numerator / denominator


def add_comparisons(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for decoder in DECODERS:
        for metric in DECODER_METRICS:
            result[f"{decoder}_near_minus_far_{metric}"] = (
                result[f"{decoder}_ring_000_016_{metric}_mean"]
                - result[f"{decoder}_ring_256_512_{metric}_mean"]
            )
    for challenger in ("marginal_map", "global_viterbi"):
        for label in ("ring_256_512", "ring_000_016", "episode"):
            result[
                f"{challenger}_minus_posterior_mean_{label}_"
                "rate_error_abs"
            ] = (
                result[f"{challenger}_{label}_rate_error_abs_mean"]
                - result[
                    f"posterior_mean_{label}_rate_error_abs_mean"
                ]
            )
    return result


def summarize_group(
    group: pd.DataFrame,
    total_sse: float,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "episodes": int(len(group)),
        "wells": int(group["well"].nunique()),
        "episode_sse_fraction": float(
            group["episode_sse"].sum() / total_sse
        ),
    }
    for decoder in DECODERS:
        for label in ("ring_256_512", "ring_000_016", "episode"):
            for metric in DECODER_METRICS:
                column = f"{decoder}_{label}_{metric}_mean"
                summary[f"{column}_episode_mean"] = float(
                    group[column].mean()
                )
                summary[f"{column}_episode_median"] = float(
                    group[column].median()
                )
                summary[f"{column}_pooled"] = pooled_interval_mean(
                    group,
                    decoder,
                    label,
                    metric,
                )
        change = f"{decoder}_near_minus_far_rate_error_abs"
        finite = group[change].notna()
        grows = finite & (group[change] > 0.0)
        summary[f"{change}_eligible_episodes"] = int(finite.sum())
        summary[f"{change}_mean"] = float(group[change].mean())
        summary[f"{change}_median"] = float(group[change].median())
        summary[f"{change}_positive_fraction_eligible"] = (
            fraction_within(grows, finite)
        )
        summary[f"{change}_positive_sse_fraction_eligible"] = (
            weighted_fraction_within(
                grows,
                finite,
                group["episode_sse"],
            )
        )
        summary[f"{change}_vs_transition_crescendo_spearman"] = (
            safe_spearman(
                group[change],
                group["truth_pre_crescendo_near_minus_far_nll"],
            )
        )
        summary[f"{change}_vs_pre128_error_slope_spearman"] = (
            safe_spearman(
                group[change],
                group["pre128_error_slope_ft_per_row"].abs(),
            )
        )
        summary[f"{change}_vs_rmse_spearman"] = safe_spearman(
            group[change],
            group["rmse_ft"],
        )
        far = pooled_interval_mean(
            group,
            decoder,
            "ring_256_512",
            "rate_error_abs",
        )
        near = pooled_interval_mean(
            group,
            decoder,
            "ring_000_016",
            "rate_error_abs",
        )
        summary[f"{decoder}_near_vs_far_rate_error_abs_ratio_pooled"] = (
            float(near / far)
            if near is not None and far is not None and far > 0.0
            else None
        )
    for challenger in ("marginal_map", "global_viterbi"):
        for label in ("ring_256_512", "ring_000_016", "episode"):
            change = (
                f"{challenger}_minus_posterior_mean_{label}_"
                "rate_error_abs"
            )
            finite = group[change].notna()
            improves = finite & (group[change] < 0.0)
            summary[f"{change}_mean"] = float(group[change].mean())
            summary[f"{change}_median"] = float(group[change].median())
            summary[f"{change}_improves_fraction_eligible"] = (
                fraction_within(improves, finite)
            )
            summary[f"{change}_improves_sse_fraction_eligible"] = (
                weighted_fraction_within(
                    improves,
                    finite,
                    group["episode_sse"],
                )
            )
            summary[f"{change}_vs_viterbi_rmse_gain_spearman"] = (
                safe_spearman(
                    -group[change],
                    group["viterbi_rmse_gain_ft"],
                )
                if challenger == "global_viterbi"
                else None
            )
    return summary


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    train_dir = resolve(root, Path("data/raw/train"))
    candidates_path = resolve(root, args.candidates)
    episodes_path = resolve(root, args.episodes)
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)

    episodes = pd.read_csv(episodes_path)
    episodes["well"] = episodes["well"].astype(str)
    episode_lookup = {
        well: group.to_dict("records")
        for well, group in episodes.groupby("well", sort=False)
    }
    records: list[dict[str, Any]] = []
    seen_wells: set[str] = set()
    maximum_posterior_mean_parity_error = 0.0
    posterior_mean_parity_checks = 0
    for well, frame in iter_well_frames(candidates_path, args.chunksize):
        specs = episode_lookup.get(well, [])
        if not specs:
            continue
        seen_wells.add(well)
        (
            horizontal,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = load_well_inputs(train_dir, well)
        row_index = pd.to_numeric(
            frame["row_idx"],
            errors="raise",
        ).to_numpy(np.int64)
        previous_index = int(row_index[0]) - 1
        md = pd.to_numeric(
            horizontal["MD"],
            errors="raise",
        ).to_numpy(np.float64)
        z = pd.to_numeric(
            horizontal["Z"],
            errors="raise",
        ).to_numpy(np.float64)
        dmd = np.maximum(
            np.diff(np.concatenate([[md[previous_index]], md[row_index]])),
            1.0,
        )
        truth = pd.to_numeric(
            frame["true_tvt_readout_only"],
            errors="raise",
        ).to_numpy(np.float64)
        previous_tvt = float(horizontal.iloc[previous_index]["TVT_input"])
        previous_u = previous_tvt + z[previous_index]
        true_rate = np.diff(
            np.concatenate([[previous_u], truth + z[row_index]])
        ) / dmd
        decoder_arrays: dict[str, dict[str, np.ndarray]] = {}
        for decoder, column in DECODERS.items():
            path = pd.to_numeric(
                frame[column],
                errors="raise",
            ).to_numpy(np.float64)
            decoded_rate = np.diff(
                np.concatenate([[previous_u], path + z[row_index]])
            ) / dmd
            rate_error = decoded_rate - true_rate
            decoder_arrays[decoder] = {
                "decoded_rate_abs": np.abs(decoded_rate),
                "rate_error_signed": rate_error,
                "rate_error_abs": np.abs(rate_error),
            }
        for spec in specs:
            record = dict(spec)
            start = int(
                np.searchsorted(
                    row_index,
                    int(spec["start_row_idx"]),
                )
            )
            end = int(
                np.searchsorted(
                    row_index,
                    int(spec["end_row_idx_exclusive"]),
                )
            )
            if end - start != int(spec["rows"]):
                raise ValueError(
                    f"{spec['episode_id']}: episode row mismatch"
                )
            for decoder, arrays in decoder_arrays.items():
                add_interval(
                    record,
                    decoder,
                    "episode",
                    start,
                    end,
                    arrays,
                )
                for label, inner, outer in RINGS:
                    ring_start = max(0, start - outer)
                    ring_end = max(0, start - inner)
                    expected_rows = int(
                        spec[f"truth_ring_{label}_rows"]
                    )
                    if ring_end - ring_start != expected_rows:
                        raise ValueError(
                            f"{spec['episode_id']}: {label} row mismatch"
                        )
                    add_interval(
                        record,
                        decoder,
                        f"ring_{label}",
                        ring_start,
                        ring_end,
                        arrays,
                    )
            for label in ("ring_256_512", "ring_000_016", "episode"):
                existing = spec[f"rate_{label}_rate_error_abs_mean"]
                recomputed = record[
                    f"posterior_mean_{label}_rate_error_abs_mean"
                ]
                if pd.isna(existing) and recomputed is None:
                    continue
                if pd.isna(existing) or recomputed is None:
                    raise ValueError(
                        f"{spec['episode_id']}: {label} parity null mismatch"
                    )
                posterior_mean_parity_checks += 1
                maximum_posterior_mean_parity_error = max(
                    maximum_posterior_mean_parity_error,
                    abs(float(existing) - float(recomputed)),
                )
            records.append(record)
    if seen_wells != set(episode_lookup):
        missing = sorted(set(episode_lookup) - seen_wells)
        raise ValueError(f"episode wells missing: {missing[:5]}")
    if maximum_posterior_mean_parity_error > 1e-12:
        raise ValueError(
            "posterior-mean rate metric parity failed: "
            f"{maximum_posterior_mean_parity_error}"
        )

    result = add_comparisons(pd.DataFrame(records))
    if len(result) != len(episodes):
        raise ValueError(
            f"episode count mismatch: {len(result)} != {len(episodes)}"
        )
    result.to_csv(
        output / "episode_decoder_rate_lag_metrics.csv",
        index=False,
    )
    total_sse = float(result["episode_sse"].sum())
    cause_rows = []
    for cause_bucket, group in result.groupby("cause_bucket", sort=True):
        cause_rows.append(
            {
                "cause_bucket": cause_bucket,
                **summarize_group(group, total_sse),
            }
        )
    pd.DataFrame(cause_rows).to_csv(
        output / "cause_decoder_rate_lag_summary.csv",
        index=False,
    )
    summary = {
        "scope": {
            "episodes": int(len(result)),
            "wells": int(result["well"].nunique()),
            "episode_rows": int(result["rows"].sum()),
        },
        "source_sha256": {
            str(candidates_path.relative_to(root)): sha256(candidates_path),
            str(episodes_path.relative_to(root)): sha256(episodes_path),
        },
        "overall": summarize_group(result, total_sse),
        "cause_bucket_summary": cause_rows,
        "guards": {
            "maximum_posterior_mean_rate_metric_parity_abs": (
                maximum_posterior_mean_parity_error
            ),
            "posterior_mean_rate_metric_parity_checks": (
                posterior_mean_parity_checks
            ),
            "global_viterbi_is_topk_path_1": True,
            "marginal_map_is_not_a_legal_joint_path": True,
            "interpretation": (
                "Frozen decoder paths and readout-only truth establish "
                "decoder-specific kinematics. They do not expose hidden "
                "filtered rate posterior or separate alpha from beta."
            ),
            "prediction_generation": False,
            "hmm_rerun": False,
            "model_or_booster": False,
        },
    }
    text = json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    )
    (output / "summary.json").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
