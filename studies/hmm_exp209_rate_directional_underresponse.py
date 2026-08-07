#!/usr/bin/env python3
"""Split frozen-decoder rate errors by direction around offset onset.

For posterior mean and legal global Viterbi trajectories, every row with
nonzero true U-rate is classified as zero-directed under-response, opposite
motion, same-direction overshoot, or an exact/tolerance tie.  This is a
truth-late readout over saved predictions; no HMM or model is rerun.
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
    "studies/hmm_exp209_decoder_rate_lag_timing_20260726/"
    "episode_decoder_rate_lag_metrics.csv"
)
DEFAULT_OUTPUT = Path(
    "studies/hmm_exp209_rate_directional_underresponse_20260726"
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
    "global_viterbi": "topk_path_1",
}
CLASSES = (
    "zero_directed_underresponse",
    "opposite_direction",
    "same_direction_overshoot",
    "tie_or_boundary",
)
RATE_TOLERANCE = 1e-9


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


def classify_rates(
    true_rate: np.ndarray,
    decoded_rate: np.ndarray,
) -> dict[str, np.ndarray]:
    moving = np.abs(true_rate) > RATE_TOLERANCE
    product = true_rate * decoded_rate
    under = (
        moving
        & (product >= -RATE_TOLERANCE**2)
        & (np.abs(decoded_rate) + RATE_TOLERANCE < np.abs(true_rate))
    )
    opposite = moving & (product < -RATE_TOLERANCE**2)
    overshoot = (
        moving
        & (product > RATE_TOLERANCE**2)
        & (np.abs(decoded_rate) > np.abs(true_rate) + RATE_TOLERANCE)
    )
    tie = moving & ~(under | opposite | overshoot)
    class_count = (
        under.astype(np.int8)
        + opposite.astype(np.int8)
        + overshoot.astype(np.int8)
        + tie.astype(np.int8)
    )
    if np.any(class_count != moving.astype(np.int8)):
        raise ValueError("direction classes do not partition moving rows")
    return {
        "moving": moving,
        "zero_directed_underresponse": under,
        "opposite_direction": opposite,
        "same_direction_overshoot": overshoot,
        "tie_or_boundary": tie,
    }


def add_interval(
    record: dict[str, Any],
    decoder: str,
    label: str,
    start: int,
    end: int,
    rate_error_abs: np.ndarray,
    classes: dict[str, np.ndarray],
) -> None:
    prefix = f"{decoder}_{label}"
    moving = classes["moving"][start:end]
    moving_rows = int(moving.sum())
    record[f"{prefix}_rows"] = int(end - start)
    record[f"{prefix}_moving_true_rate_rows"] = moving_rows
    record[f"{prefix}_all_rate_error_abs_sum"] = float(
        rate_error_abs[start:end].sum()
    )
    total_error = float(rate_error_abs[start:end][moving].sum())
    record[f"{prefix}_moving_rate_error_abs_sum"] = total_error
    for name in CLASSES:
        mask = classes[name][start:end]
        count = int(mask.sum())
        error_sum = float(rate_error_abs[start:end][mask].sum())
        record[f"{prefix}_{name}_rows"] = count
        record[f"{prefix}_{name}_fraction"] = (
            float(count / moving_rows) if moving_rows else None
        )
        record[f"{prefix}_{name}_rate_error_abs_sum"] = error_sum
        record[f"{prefix}_{name}_rate_error_abs_share"] = (
            float(error_sum / total_error) if total_error > 0.0 else None
        )
    class_rows = sum(
        record[f"{prefix}_{name}_rows"] for name in CLASSES
    )
    if class_rows != moving_rows:
        raise ValueError(f"{prefix}: class row count mismatch")


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
            prefix = f"{decoder}_{label}"
            moving_rows = int(group[f"{prefix}_moving_true_rate_rows"].sum())
            total_error = float(
                group[f"{prefix}_moving_rate_error_abs_sum"].sum()
            )
            summary[f"{prefix}_moving_true_rate_rows"] = moving_rows
            for name in CLASSES:
                rows = int(group[f"{prefix}_{name}_rows"].sum())
                error_sum = float(
                    group[
                        f"{prefix}_{name}_rate_error_abs_sum"
                    ].sum()
                )
                summary[f"{prefix}_{name}_fraction_episode_mean"] = float(
                    group[f"{prefix}_{name}_fraction"].mean()
                )
                summary[f"{prefix}_{name}_fraction_pooled"] = (
                    float(rows / moving_rows) if moving_rows else None
                )
                summary[
                    f"{prefix}_{name}_rate_error_abs_share_pooled"
                ] = (
                    float(error_sum / total_error)
                    if total_error > 0.0
                    else None
                )
        near = (
            f"{decoder}_ring_000_016_"
            "zero_directed_underresponse_fraction"
        )
        far = (
            f"{decoder}_ring_256_512_"
            "zero_directed_underresponse_fraction"
        )
        change = (
            f"{decoder}_near_minus_far_"
            "zero_directed_underresponse_fraction"
        )
        values = group[near] - group[far]
        finite = values.notna()
        increases = finite & (values > 0.0)
        summary[f"{change}_eligible_episodes"] = int(finite.sum())
        summary[f"{change}_mean"] = float(values.mean())
        summary[f"{change}_median"] = float(values.median())
        summary[f"{change}_positive_fraction_eligible"] = (
            float(increases.sum() / finite.sum()) if finite.any() else None
        )
        denominator = float(group.loc[finite, "episode_sse"].sum())
        summary[f"{change}_positive_sse_fraction_eligible"] = (
            float(group.loc[increases, "episode_sse"].sum() / denominator)
            if denominator > 0.0
            else None
        )
        summary[f"{change}_vs_transition_crescendo_spearman"] = (
            safe_spearman(
                values,
                group["truth_pre_crescendo_near_minus_far_nll"],
            )
        )
        summary[f"{change}_vs_rate_error_growth_spearman"] = (
            safe_spearman(
                values,
                group[
                    f"{decoder}_near_minus_far_rate_error_abs"
                ],
            )
        )
        summary[f"{change}_vs_pre128_error_slope_spearman"] = (
            safe_spearman(
                values,
                group["pre128_error_slope_ft_per_row"].abs(),
            )
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
    maximum_posterior_mean_rate_error_parity = 0.0
    parity_checks = 0
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
        previous_u = (
            float(horizontal.iloc[previous_index]["TVT_input"])
            + z[previous_index]
        )
        true_rate = np.diff(
            np.concatenate([[previous_u], truth + z[row_index]])
        ) / dmd
        decoder_data: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
        for decoder, column in DECODERS.items():
            path = pd.to_numeric(
                frame[column],
                errors="raise",
            ).to_numpy(np.float64)
            decoded_rate = np.diff(
                np.concatenate([[previous_u], path + z[row_index]])
            ) / dmd
            rate_error_abs = np.abs(decoded_rate - true_rate)
            decoder_data[decoder] = (
                rate_error_abs,
                classify_rates(true_rate, decoded_rate),
            )
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
            for decoder, (rate_error_abs, classes) in decoder_data.items():
                add_interval(
                    record,
                    decoder,
                    "episode",
                    start,
                    end,
                    rate_error_abs,
                    classes,
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
                        rate_error_abs,
                        classes,
                    )
            for label in ("ring_256_512", "ring_000_016", "episode"):
                existing = spec[
                    f"posterior_mean_{label}_rate_error_abs_mean"
                ]
                interval_rows = record[f"posterior_mean_{label}_rows"]
                all_sum = record[
                    f"posterior_mean_{label}_all_rate_error_abs_sum"
                ]
                recomputed = (
                    all_sum / interval_rows if interval_rows else None
                )
                if pd.isna(existing) and recomputed is None:
                    continue
                if pd.isna(existing) or recomputed is None:
                    raise ValueError(
                        f"{spec['episode_id']}: {label} parity null mismatch"
                    )
                parity_checks += 1
                maximum_posterior_mean_rate_error_parity = max(
                    maximum_posterior_mean_rate_error_parity,
                    abs(float(existing) - float(recomputed)),
                )
            records.append(record)
    if seen_wells != set(episode_lookup):
        missing = sorted(set(episode_lookup) - seen_wells)
        raise ValueError(f"episode wells missing: {missing[:5]}")
    if maximum_posterior_mean_rate_error_parity > 1e-12:
        raise ValueError(
            "posterior-mean rate error parity failed: "
            f"{maximum_posterior_mean_rate_error_parity}"
        )

    result = pd.DataFrame(records)
    if len(result) != len(episodes):
        raise ValueError(
            f"episode count mismatch: {len(result)} != {len(episodes)}"
        )
    result.to_csv(
        output / "episode_directional_rate_metrics.csv",
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
        output / "cause_directional_rate_summary.csv",
        index=False,
    )
    summary = {
        "scope": {
            "episodes": int(len(result)),
            "wells": int(result["well"].nunique()),
            "episode_rows": int(result["rows"].sum()),
        },
        "definition": {
            "rate_tolerance": RATE_TOLERANCE,
            "zero_directed_underresponse": (
                "true rate is nonzero, decoded rate is not opposite, "
                "and abs(decoded rate) < abs(true rate)"
            ),
            "classes": list(CLASSES),
        },
        "source_sha256": {
            str(candidates_path.relative_to(root)): sha256(candidates_path),
            str(episodes_path.relative_to(root)): sha256(episodes_path),
        },
        "overall": summarize_group(result, total_sse),
        "cause_bucket_summary": cause_rows,
        "guards": {
            "maximum_posterior_mean_rate_error_parity_abs": (
                maximum_posterior_mean_rate_error_parity
            ),
            "posterior_mean_rate_error_parity_checks": parity_checks,
            "direction_classes_partition_moving_rows": True,
            "episode_zero_true_rate_rows": int(
                (
                    result["posterior_mean_episode_rows"]
                    - result[
                        "posterior_mean_episode_moving_true_rate_rows"
                    ]
                ).sum()
            ),
            "global_viterbi_is_legal_joint_path": True,
            "interpretation": (
                "Direction classes use frozen decoder paths and "
                "readout-only truth. They do not reveal the hidden rate "
                "state posterior or isolate alpha from beta."
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
