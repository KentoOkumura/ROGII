#!/usr/bin/env python3
"""Audit GR-emission barriers between exp209 candidate and truth paths.

For each persistent episode, evaluate two 41-point slices from the already
frozen posterior-mean path:

    truth morph: candidate + f * (truth - candidate)
    datum shift: candidate - f * mean(candidate - truth)

The exp209 Gaussian/capped typewell-GR NLL is reconstructed on observed rows,
all interpolated rows, and missing/imputed rows. A non-monotone peak above the
candidate endpoint is an emission barrier that can help keep a
translation-invariant HMM in the wrong datum basin. The constant datum shift
isolates the translation gauge; the pointwise truth morph also removes
within-episode shape error.

These are truth-late diagnostic slices, not HMM paths, prediction candidates,
or claims about the minimum-energy sequential recovery path.
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
    "studies/hmm_exp209_offset_cause_readout_20260725/"
    "persistent_offset_episodes.csv"
)
DEFAULT_PRIOR_EPISODES = Path(
    "studies/hmm_exp209_actual_geometry_transition_prior_20260726/"
    "episode_metrics.csv"
)
DEFAULT_OUTPUT = Path(
    "studies/hmm_exp209_gr_rigid_shift_barrier_20260726"
)
FRACTIONS = np.linspace(0.0, 1.0, 41, dtype=np.float64)
NLL_CAP = 600.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument(
        "--prior-episodes",
        type=Path,
        default=DEFAULT_PRIOR_EPISODES,
    )
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


def safe_spearman(left: pd.Series, right: pd.Series) -> float:
    finite = np.isfinite(left) & np.isfinite(right)
    if int(np.sum(finite)) < 3:
        return float("nan")
    x = left.loc[finite].to_numpy(np.float64)
    y = right.loc[finite].to_numpy(np.float64)
    if np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def weighted_fraction(mask: pd.Series, weight: pd.Series) -> float:
    weights = weight.to_numpy(np.float64)
    return float(np.sum(weights[mask.to_numpy(bool)]) / np.sum(weights))


def cause_bucket(
    observed_class: str,
    prior_matches_actual: bool,
) -> str:
    mapping = {
        (True, "candidate_strong"): "prior_and_wrong_gr_reinforce",
        (True, "near_tie"): "prior_aligned_gr_ambiguous",
        (True, "truth_strong"): "prior_persists_against_truth_gr",
        (False, "candidate_strong"): "candidate_gr_with_opposed_prior",
        (False, "near_tie"): "opposed_prior_gr_ambiguous",
        (False, "truth_strong"): "neither_prior_nor_observed_gr",
    }
    return mapping[(prior_matches_actual, observed_class)]


def nll_curve(
    observed_gr: np.ndarray,
    candidate: np.ndarray,
    truth: np.ndarray,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
    sigma: float,
) -> np.ndarray:
    if len(observed_gr) == 0:
        return np.full(len(FRACTIONS), np.nan, dtype=np.float64)
    curve = np.empty(len(FRACTIONS), dtype=np.float64)
    for index, fraction in enumerate(FRACTIONS):
        path = candidate + fraction * (truth - candidate)
        reference = np.interp(path, typewell_tvt, typewell_gr)
        nll = 0.5 * np.minimum(
            ((observed_gr - reference) / sigma) ** 2,
            NLL_CAP,
        )
        curve[index] = float(np.sum(nll))
    return curve


def curve_metrics(
    curve: np.ndarray,
    rows: int,
    prefix: str,
    target_label: str,
) -> dict[str, Any]:
    if rows == 0 or not np.all(np.isfinite(curve)):
        return {
            f"{prefix}_rows": int(rows),
            f"{prefix}_candidate_nll": float("nan"),
            f"{prefix}_{target_label}_nll": float("nan"),
            f"{prefix}_{target_label}_gain_nll": float("nan"),
            f"{prefix}_barrier_above_candidate_nll": float("nan"),
            f"{prefix}_barrier_above_candidate_nll_per_row": float("nan"),
            f"{prefix}_barrier_above_chord_nll": float("nan"),
            f"{prefix}_barrier_fraction": float("nan"),
            f"{prefix}_monotone_to_{target_label}": False,
        }
    candidate_nll = float(curve[0])
    target_nll = float(curve[-1])
    maximum_index = int(np.argmax(curve))
    chord = (
        (1.0 - FRACTIONS) * candidate_nll
        + FRACTIONS * target_nll
    )
    tolerance = 1e-12 * max(1.0, float(np.max(np.abs(curve))))
    return {
        f"{prefix}_rows": int(rows),
        f"{prefix}_candidate_nll": candidate_nll,
        f"{prefix}_{target_label}_nll": target_nll,
        f"{prefix}_{target_label}_gain_nll": candidate_nll - target_nll,
        f"{prefix}_barrier_above_candidate_nll": float(
            np.max(curve) - candidate_nll
        ),
        f"{prefix}_barrier_above_candidate_nll_per_row": float(
            (np.max(curve) - candidate_nll) / rows
        ),
        f"{prefix}_barrier_above_chord_nll": float(
            np.max(curve - chord)
        ),
        f"{prefix}_barrier_fraction": float(FRACTIONS[maximum_index]),
        f"{prefix}_monotone_to_{target_label}": bool(
            np.all(np.diff(curve) <= tolerance)
        ),
    }


def summarize_group(
    group: pd.DataFrame,
    total_sse: float,
    prefix: str,
    target_label: str,
) -> dict[str, Any]:
    barrier = group[f"{prefix}_barrier_above_candidate_nll"]
    per_row = group[
        f"{prefix}_barrier_above_candidate_nll_per_row"
    ]
    target_better = (
        group[f"{prefix}_{target_label}_gain_nll"] > 0.0
    )
    barrier_gt5 = barrier > 5.0
    barrier_gt20 = barrier > 20.0
    monotone = group[f"{prefix}_monotone_to_{target_label}"]
    better_group = group.loc[target_better]
    better_barrier = barrier.loc[target_better]
    better_weight = better_group["episode_sse"]

    def better_fraction(mask: pd.Series) -> float | None:
        if len(better_group) == 0:
            return None
        return float(mask.loc[target_better].mean())

    def better_weighted_fraction(mask: pd.Series) -> float | None:
        if len(better_group) == 0:
            return None
        return weighted_fraction(
            mask.loc[target_better],
            better_weight,
        )

    return {
        "episodes": int(len(group)),
        "wells": int(group["well"].nunique()),
        "episode_sse_fraction": float(
            group["episode_sse"].sum() / total_sse
        ),
        f"{prefix}_{target_label}_better_fraction": float(
            target_better.mean()
        ),
        f"{prefix}_{target_label}_better_sse_fraction": weighted_fraction(
            target_better,
            group["episode_sse"],
        ),
        f"{prefix}_barrier_total_median_nll_given_{target_label}_better": (
            float(better_barrier.median())
            if len(better_group) > 0
            else None
        ),
        f"{prefix}_barrier_gt5_fraction_given_{target_label}_better": (
            better_fraction(barrier_gt5)
        ),
        (
            f"{prefix}_barrier_gt5_sse_fraction_given_"
            f"{target_label}_better"
        ): better_weighted_fraction(barrier_gt5),
        f"{prefix}_barrier_gt20_fraction_given_{target_label}_better": (
            better_fraction(barrier_gt20)
        ),
        (
            f"{prefix}_barrier_gt20_sse_fraction_given_"
            f"{target_label}_better"
        ): better_weighted_fraction(barrier_gt20),
        f"{prefix}_barrier_total_median_nll": float(barrier.median()),
        f"{prefix}_barrier_total_p90_nll": float(
            barrier.quantile(0.9)
        ),
        f"{prefix}_barrier_per_row_median_nll": float(
            per_row.median()
        ),
        f"{prefix}_barrier_gt5_fraction": float(barrier_gt5.mean()),
        f"{prefix}_barrier_gt5_sse_fraction": weighted_fraction(
            barrier_gt5,
            group["episode_sse"],
        ),
        f"{prefix}_barrier_gt20_fraction": float(
            barrier_gt20.mean()
        ),
        f"{prefix}_barrier_gt20_sse_fraction": weighted_fraction(
            barrier_gt20,
            group["episode_sse"],
        ),
        f"{prefix}_monotone_to_{target_label}_fraction": float(
            monotone.mean()
        ),
        f"{prefix}_barrier_vs_actual_rmse_spearman": safe_spearman(
            barrier,
            group["actual_rmse_ft"],
        ),
        f"{prefix}_barrier_vs_viterbi_gain_spearman": safe_spearman(
            barrier,
            group["viterbi_rmse_gain_ft"],
        ),
    }


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    candidates_path = resolve(root, args.candidates)
    episodes_path = resolve(root, args.episodes)
    prior_path = resolve(root, args.prior_episodes)
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)

    episodes = pd.read_csv(episodes_path)
    prior = pd.read_csv(prior_path)
    key = ["episode_id", "well"]
    if episodes.duplicated(key).any() or prior.duplicated(key).any():
        raise ValueError("episode inputs contain duplicate keys")
    audit = episodes.merge(
        prior[
            [
                "episode_id",
                "well",
                "actual_mean_error_ft",
                "current_mean_error_ft",
                "actual_rmse_ft",
            ]
        ],
        on=key,
        validate="one_to_one",
    )
    if len(audit) != len(episodes) or len(audit) != len(prior):
        raise ValueError("episode key sets differ")
    audit["current_prior_sign_matches_actual"] = (
        np.sign(audit["current_mean_error_ft"])
        == np.sign(audit["actual_mean_error_ft"])
    )
    audit["cause_bucket"] = [
        cause_bucket(str(observed_class), bool(prior_match))
        for observed_class, prior_match in audit[
            [
                "observed_emission_evidence_class",
                "current_prior_sign_matches_actual",
            ]
        ].itertuples(index=False, name=None)
    ]
    episode_lookup = {
        str(well): group.sort_values("start_row_idx").to_dict("records")
        for well, group in audit.groupby("well", sort=False)
    }

    result_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for well, frame in iter_well_frames(
        candidates_path,
        int(args.chunksize),
    ):
        specs = episode_lookup.get(str(well))
        if specs is None:
            continue
        (
            horizontal,
            typewell_tvt,
            typewell_gr,
            sigma,
            _,
            _,
            _,
            _,
        ) = load_well_inputs(root / "data/raw/train", str(well))
        row_index = pd.to_numeric(
            frame["row_idx"],
            errors="raise",
        ).to_numpy(np.int64)
        raw_gr_full = pd.to_numeric(
            horizontal["GR"],
            errors="coerce",
        )
        filled_gr_full = (
            raw_gr_full.interpolate(limit_direction="both")
            .fillna(float(np.nanmean(typewell_gr)))
            .to_numpy(np.float64)
        )
        for spec in specs:
            mask = (
                (row_index >= int(spec["start_row_idx"]))
                & (row_index < int(spec["end_row_idx_exclusive"]))
            )
            if int(np.sum(mask)) != int(spec["rows"]):
                raise ValueError(
                    f"{spec['episode_id']}: candidate rows differ"
                )
            rows = row_index[mask]
            candidate = pd.to_numeric(
                frame.loc[mask, "posterior_mean"],
                errors="raise",
            ).to_numpy(np.float64)
            truth = pd.to_numeric(
                frame.loc[mask, "true_tvt_readout_only"],
                errors="raise",
            ).to_numpy(np.float64)
            raw_gr = raw_gr_full.iloc[rows].to_numpy(np.float64)
            filled_gr = filled_gr_full[rows]
            observed = np.isfinite(raw_gr)
            missing = ~observed
            targets = {
                "truth_morph": {
                    "target": truth,
                    "metric_prefix": "",
                    "target_label": "truth",
                },
                "constant_datum_shift": {
                    "target": candidate
                    - float(spec["actual_mean_error_ft"]),
                    "metric_prefix": "datum_",
                    "target_label": "corrected",
                },
            }
            record: dict[str, Any] = {
                "episode_id": spec["episode_id"],
                "well": str(well),
                "rows": int(spec["rows"]),
            }
            for path_kind, target_spec in targets.items():
                target = target_spec["target"]
                curves = {
                    "observed": nll_curve(
                        raw_gr[observed],
                        candidate[observed],
                        target[observed],
                        typewell_tvt,
                        typewell_gr,
                        sigma,
                    ),
                    "all": nll_curve(
                        filled_gr,
                        candidate,
                        target,
                        typewell_tvt,
                        typewell_gr,
                        sigma,
                    ),
                    "imputed": nll_curve(
                        filled_gr[missing],
                        candidate[missing],
                        target[missing],
                        typewell_tvt,
                        typewell_gr,
                        sigma,
                    ),
                }
                for scope, curve in curves.items():
                    curve_count = (
                        int(np.sum(observed))
                        if scope == "observed"
                        else int(np.sum(missing))
                        if scope == "imputed"
                        else len(rows)
                    )
                    metric_prefix = (
                        f"{target_spec['metric_prefix']}{scope}"
                    )
                    record.update(
                        curve_metrics(
                            curve,
                            curve_count,
                            metric_prefix,
                            str(target_spec["target_label"]),
                        )
                    )
                    for fraction, nll in zip(
                        FRACTIONS,
                        curve,
                        strict=True,
                    ):
                        curve_rows.append(
                            {
                                "episode_id": spec["episode_id"],
                                "well": str(well),
                                "path_kind": path_kind,
                                "scope": scope,
                                "fraction_candidate_to_target": float(
                                    fraction
                                ),
                                "nll": float(nll),
                                "rows": curve_count,
                            }
                        )
            result_rows.append(record)

    landscape = pd.DataFrame(result_rows)
    if len(landscape) != len(audit):
        raise ValueError(
            f"landscape episodes {len(landscape)} != {len(audit)}"
        )
    landscape = audit.merge(
        landscape,
        on=key + ["rows"],
        validate="one_to_one",
    )
    observed_parity = float(
        np.max(
            np.abs(
                (
                    landscape["observed_truth_nll"]
                    - landscape["observed_candidate_nll"]
                )
                - landscape[
                    "observed_total_truth_minus_candidate_nll"
                ]
            )
        )
    )
    if observed_parity > 1e-9:
        raise ValueError(
            f"observed endpoint NLL parity failed: {observed_parity}"
        )

    landscape.to_csv(output / "episode_landscape.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(
        output / "episode_fraction_curves.csv.gz",
        index=False,
        compression={
            "method": "gzip",
            "compresslevel": 6,
            "mtime": 0,
        },
    )
    total_sse = float(landscape["episode_sse"].sum())
    group_rows = []
    for bucket, group in landscape.groupby("cause_bucket", sort=True):
        group_rows.append(
            {
                "cause_bucket": bucket,
                **summarize_group(
                    group,
                    total_sse,
                    "observed",
                    "truth",
                ),
                **summarize_group(
                    group,
                    total_sse,
                    "all",
                    "truth",
                ),
                **summarize_group(
                    group,
                    total_sse,
                    "imputed",
                    "truth",
                ),
                **summarize_group(
                    group,
                    total_sse,
                    "datum_observed",
                    "corrected",
                ),
                **summarize_group(
                    group,
                    total_sse,
                    "datum_all",
                    "corrected",
                ),
                **summarize_group(
                    group,
                    total_sse,
                    "datum_imputed",
                    "corrected",
                ),
            }
        )
    group_frame = pd.DataFrame(group_rows)
    group_frame.to_csv(
        output / "cause_bucket_barrier_summary.csv",
        index=False,
    )

    summary = {
        "scope": {
            "episodes": int(len(landscape)),
            "wells": int(landscape["well"].nunique()),
            "fractions": [float(value) for value in FRACTIONS],
            "observed_endpoint_nll_parity_max_abs": observed_parity,
        },
        "source_sha256": {
            str(candidates_path.relative_to(root)): sha256(
                candidates_path
            ),
            str(episodes_path.relative_to(root)): sha256(episodes_path),
            str(prior_path.relative_to(root)): sha256(prior_path),
        },
        "overall": {
            **summarize_group(
                landscape,
                total_sse,
                "observed",
                "truth",
            ),
            **summarize_group(
                landscape,
                total_sse,
                "all",
                "truth",
            ),
            **summarize_group(
                landscape,
                total_sse,
                "imputed",
                "truth",
            ),
            **summarize_group(
                landscape,
                total_sse,
                "datum_observed",
                "corrected",
            ),
            **summarize_group(
                landscape,
                total_sse,
                "datum_all",
                "corrected",
            ),
            **summarize_group(
                landscape,
                total_sse,
                "datum_imputed",
                "corrected",
            ),
        },
        "cause_bucket_summary": group_rows,
        "guards": {
            "truth_usage": (
                "Posterior-mean candidates and keys are loaded from the "
                "already frozen exp270 artifact before truth is used to "
                "define the diagnostic morph."
            ),
            "interpretation": (
                "The 41-point pointwise-truth and constant-datum slices are "
                "not sequential HMM recovery paths or minimum-action "
                "barriers. Positive barriers are sufficient evidence of "
                "emission non-convexity along the corresponding slice; zero "
                "barriers do not rule out a dynamic transition or "
                "joint-state barrier."
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
