#!/usr/bin/env python3
"""Convert nested pre-onset grammar windows into non-overlapping time rings."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DEFAULT_EPISODES = Path(
    "studies/hmm_exp209_truth_path_grammar_audit_20260726/"
    "episode_grammar_metrics.csv"
)
DEFAULT_GRAMMAR_SUMMARY = Path(
    "studies/hmm_exp209_truth_path_grammar_audit_20260726/"
    "summary.json"
)
DEFAULT_OUTPUT = Path(
    "studies/hmm_exp209_truth_grammar_temporal_readout_20260726"
)
NESTED_WINDOWS = (16, 64, 128, 256, 512)
RINGS = (
    ("000_016", 0, 16),
    ("016_064", 16, 64),
    ("064_128", 64, 128),
    ("128_256", 128, 256),
    ("256_512", 256, 512),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument(
        "--grammar-summary",
        type=Path,
        default=DEFAULT_GRAMMAR_SUMMARY,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
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


def weighted_fraction(mask: pd.Series, weight: pd.Series) -> float | None:
    total = float(weight.sum())
    if total <= 0.0:
        return None
    return float(weight.loc[mask].sum() / total)


def weighted_fraction_within(
    mask: pd.Series,
    eligible: pd.Series,
    weight: pd.Series,
) -> float | None:
    total = float(weight.loc[eligible].sum())
    if total <= 0.0:
        return None
    return float(weight.loc[mask & eligible].sum() / total)


def finite_float(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def safe_spearman(left: pd.Series, right: pd.Series) -> float | None:
    finite = np.isfinite(left) & np.isfinite(right)
    if int(finite.sum()) < 3:
        return None
    x = left.loc[finite].to_numpy(np.float64)
    y = right.loc[finite].to_numpy(np.float64)
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return None
    return float(spearmanr(x, y).statistic)


def nested_sum(frame: pd.DataFrame, window: int) -> pd.Series:
    return (
        frame[
            f"truth_pre{window}_conditioned_transition_nll_mean"
        ].fillna(0.0)
        * frame[f"truth_pre{window}_rows"]
    )


def add_ring_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for label, inner, outer in RINGS:
        outer_rows = result[f"truth_pre{outer}_rows"]
        outer_sum = nested_sum(result, outer)
        if inner == 0:
            ring_rows = outer_rows
            ring_sum = outer_sum
        else:
            inner_rows = result[f"truth_pre{inner}_rows"]
            ring_rows = outer_rows - inner_rows
            ring_sum = outer_sum - nested_sum(result, inner)
        result[f"truth_ring_{label}_rows"] = ring_rows.astype(np.int64)
        result[f"truth_ring_{label}_transition_nll_mean"] = np.where(
            ring_rows > 0,
            ring_sum / ring_rows,
            np.nan,
        )
    near = result["truth_ring_000_016_transition_nll_mean"]
    far = result["truth_ring_256_512_transition_nll_mean"]
    result["truth_pre_crescendo_near_minus_far_nll"] = near - far
    ring_columns = [
        f"truth_ring_{label}_transition_nll_mean"
        for label, _, _ in reversed(RINGS)
    ]
    ring_values = result[ring_columns].to_numpy(np.float64)
    crescendo_rho = np.full(len(result), np.nan, dtype=np.float64)
    strict_increase = np.zeros(len(result), dtype=bool)
    for index, values in enumerate(ring_values):
        if np.all(np.isfinite(values)):
            crescendo_rho[index] = float(
                spearmanr(
                    np.arange(len(values), dtype=np.float64),
                    values,
                ).statistic
            )
            strict_increase[index] = bool(np.all(np.diff(values) > 0.0))
    result["truth_pre_ring_crescendo_spearman"] = crescendo_rho
    result["truth_pre_ring_crescendo_eligible"] = np.all(
        np.isfinite(ring_values),
        axis=1,
    )
    result["truth_pre_ring_strict_increase"] = strict_increase
    return result


def summarize_group(
    group: pd.DataFrame,
    total_sse: float,
    nonpersistent_truth_nll: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "episodes": int(len(group)),
        "wells": int(group["well"].nunique()),
        "episode_sse_fraction": float(
            group["episode_sse"].sum() / total_sse
        ),
        "nonpersistent_truth_two_step_nll": nonpersistent_truth_nll,
    }
    for label, _, _ in RINGS:
        value_column = f"truth_ring_{label}_transition_nll_mean"
        row_column = f"truth_ring_{label}_rows"
        finite = group[value_column].notna()
        pooled_rows = int(group.loc[finite, row_column].sum())
        pooled_sum = float(
            (
                group.loc[finite, value_column]
                * group.loc[finite, row_column]
            ).sum()
        )
        result.update(
            {
                f"{value_column}_episodes": int(finite.sum()),
                f"{value_column}_episode_mean": finite_float(
                    group[value_column].mean()
                ),
                f"{value_column}_episode_median": finite_float(
                    group[value_column].median()
                ),
                f"{value_column}_pooled": (
                    float(pooled_sum / pooled_rows)
                    if pooled_rows
                    else None
                ),
                f"{value_column}_vs_rmse_spearman": safe_spearman(
                    group[value_column],
                    group["rmse_ft"],
                ),
            }
        )
    crescendo = group["truth_pre_crescendo_near_minus_far_nll"]
    finite_crescendo = crescendo.notna()
    positive = finite_crescendo & (crescendo > 0.0)
    rho = group["truth_pre_ring_crescendo_spearman"]
    finite_rho = rho.notna()
    ring_eligible = group["truth_pre_ring_crescendo_eligible"]
    strict = group["truth_pre_ring_strict_increase"]
    result.update(
        {
            "truth_pre_crescendo_eligible_episodes": int(
                finite_crescendo.sum()
            ),
            "truth_pre_crescendo_near_minus_far_nll_mean": finite_float(
                crescendo.mean()
            ),
            "truth_pre_crescendo_near_minus_far_nll_median": finite_float(
                crescendo.median()
            ),
            "truth_pre_crescendo_positive_fraction": float(
                positive.sum() / max(1, int(finite_crescendo.sum()))
            ),
            "truth_pre_crescendo_positive_sse_fraction_all_group": (
                weighted_fraction(
                    positive,
                    group["episode_sse"],
                )
            ),
            "truth_pre_crescendo_positive_sse_fraction_eligible": (
                weighted_fraction_within(
                    positive,
                    finite_crescendo,
                    group["episode_sse"],
                )
            ),
            "truth_pre_crescendo_spearman_mean": finite_float(rho.mean()),
            "truth_pre_crescendo_spearman_median": finite_float(
                rho.median()
            ),
            "truth_pre_crescendo_positive_rho_fraction": float(
                (rho.loc[finite_rho] > 0.0).mean()
            )
            if finite_rho.any()
            else None,
            "truth_pre_crescendo_ring_eligible_episodes": int(
                ring_eligible.sum()
            ),
            "truth_pre_crescendo_strict_increase_fraction": float(
                strict.loc[ring_eligible].mean()
            )
            if ring_eligible.any()
            else None,
            "truth_pre_crescendo_strict_increase_sse_fraction_eligible": (
                weighted_fraction_within(
                    strict,
                    ring_eligible,
                    group["episode_sse"],
                )
            ),
            "truth_pre_crescendo_vs_rmse_spearman": safe_spearman(
                crescendo,
                group["rmse_ft"],
            ),
            "truth_pre_crescendo_vs_pre128_error_slope_spearman": (
                safe_spearman(
                    crescendo,
                    group["pre128_error_slope_ft_per_row"].abs(),
                )
            ),
            "truth_pre_crescendo_vs_viterbi_gain_spearman": (
                safe_spearman(
                    crescendo,
                    group["viterbi_rmse_gain_ft"],
                )
            ),
        }
    )
    return result


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    episodes_path = resolve(root, args.episodes)
    grammar_summary_path = resolve(root, args.grammar_summary)
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)

    episodes = pd.read_csv(episodes_path)
    grammar_summary = json.loads(grammar_summary_path.read_text())
    nonpersistent_truth_nll = float(
        grammar_summary["row_summary"]["truth"]["nonpersistent"][
            "conditioned_transition_nll_mean"
        ]
    )
    temporal = add_ring_metrics(episodes)
    temporal.to_csv(
        output / "episode_temporal_grammar_metrics.csv",
        index=False,
    )
    total_sse = float(temporal["episode_sse"].sum())
    cause_rows = []
    for cause_bucket, group in temporal.groupby(
        "cause_bucket",
        sort=True,
    ):
        cause_rows.append(
            {
                "cause_bucket": cause_bucket,
                **summarize_group(
                    group,
                    total_sse,
                    nonpersistent_truth_nll,
                ),
            }
        )
    pd.DataFrame(cause_rows).to_csv(
        output / "cause_temporal_grammar_summary.csv",
        index=False,
    )
    summary = {
        "scope": {
            "episodes": int(len(temporal)),
            "wells": int(temporal["well"].nunique()),
            "rings_rows_before_onset": [
                {"label": label, "inner": inner, "outer": outer}
                for label, inner, outer in RINGS
            ],
        },
        "source_sha256": {
            str(episodes_path.relative_to(root)): sha256(episodes_path),
            str(grammar_summary_path.relative_to(root)): sha256(
                grammar_summary_path
            ),
        },
        "overall": summarize_group(
            temporal,
            total_sse,
            nonpersistent_truth_nll,
        ),
        "cause_bucket_summary": cause_rows,
        "guards": {
            "derivation": (
                "Non-overlapping ring sums are exact differences of nested "
                "row-count-weighted means from the frozen grammar audit."
            ),
            "interpretation": (
                "Rings are truth-late diagnostics around error-defined onset, "
                "not prediction features or causal interventions."
            ),
            "prediction_generation": False,
            "hmm_rerun": False,
            "model_or_booster": False,
        },
    }
    (output / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
