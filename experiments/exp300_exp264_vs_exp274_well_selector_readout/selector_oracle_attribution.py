from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from selector_switch_readout import (
    ARTIFACT_DIR,
    CANDIDATES,
    EXPECTED_HARD_PRIMARY_RMSE,
    EXPECTED_LONG_ROWS,
    EXPECTED_ROWS,
    EXPECTED_SCORE_SHA256,
    EXPECTED_WELLS,
    EXP264_VIEWER_PATH,
    EXP274_OOF_PATH,
    PRIMARY_COUNT,
    SCORE_PATH,
    WELL_PATH,
    load_base_arrays,
    rmse,
    row_positions_from_score_chunk,
    sha256_file,
)


DISTANCE_BINS = [-np.inf, 50.0, 100.0, 250.0, 500.0, 1000.0, np.inf]
DISTANCE_LABELS = ["0--50", "50--100", "100--250", "250--500", "500--1000", "1000+"]
TIE_TOLERANCE_FT = 1e-5


def reconstruct_selected_and_oracle(
    base: dict[str, Any],
) -> dict[str, np.ndarray]:
    selected_code = np.full(EXPECTED_ROWS, -1, np.int8)
    oracle_code = np.full(EXPECTED_ROWS, -1, np.int8)
    selected_tvt = np.full(EXPECTED_ROWS, np.nan, np.float32)
    oracle_tvt = np.full(EXPECTED_ROWS, np.nan, np.float32)
    selected_abs_error = np.full(EXPECTED_ROWS, np.nan, np.float32)
    oracle_abs_error = np.full(EXPECTED_ROWS, np.nan, np.float32)
    md_since = np.full(EXPECTED_ROWS, np.nan, np.float32)
    covered = np.zeros(EXPECTED_ROWS, bool)

    parquet = pq.ParquetFile(SCORE_PATH)
    if parquet.metadata.num_rows != EXPECTED_LONG_ROWS:
        raise ValueError("Stage C candidate-long row count changed")
    columns = [
        "well",
        "well_row_idx",
        "md_since",
        "candidate_id",
        "candidate_tvt",
        "pred_abs_error",
        "actual_abs_error",
    ]
    processed = 0
    for row_group in range(parquet.num_row_groups):
        chunk = parquet.read_row_group(row_group, columns=columns).to_pandas()
        if len(chunk) % len(CANDIDATES) != 0:
            raise ValueError("row group breaks candidate blocks")
        block_rows = len(chunk) // len(CANDIDATES)
        positions, _ = row_positions_from_score_chunk(chunk, block_rows, base)
        if bool(covered[positions].any()) or len(np.unique(positions)) != block_rows:
            raise ValueError("candidate score coverage overlaps")

        candidate_tvt = chunk["candidate_tvt"].to_numpy(np.float32).reshape(
            block_rows, len(CANDIDATES)
        )[:, :PRIMARY_COUNT]
        predicted_error = chunk["pred_abs_error"].to_numpy(np.float32).reshape(
            block_rows, len(CANDIDATES)
        )[:, :PRIMARY_COUNT]
        actual_error = chunk["actual_abs_error"].to_numpy(np.float32).reshape(
            block_rows, len(CANDIDATES)
        )[:, :PRIMARY_COUNT]
        if not (
            np.isfinite(candidate_tvt).all()
            and np.isfinite(predicted_error).all()
            and np.isfinite(actual_error).all()
        ):
            raise ValueError("primary candidate surface contains non-finite values")

        selected = np.argmin(predicted_error, axis=1).astype(np.int8)
        oracle = np.argmin(actual_error, axis=1).astype(np.int8)
        local = np.arange(block_rows)
        selected_code[positions] = selected
        oracle_code[positions] = oracle
        selected_tvt[positions] = candidate_tvt[local, selected]
        oracle_tvt[positions] = candidate_tvt[local, oracle]
        selected_abs_error[positions] = actual_error[local, selected]
        oracle_abs_error[positions] = actual_error[local, oracle]

        distance = chunk["md_since"].to_numpy(np.float32).reshape(
            block_rows, len(CANDIDATES)
        )
        if not np.all(distance == distance[:, :1]):
            raise ValueError("md_since changed within candidate block")
        md_since[positions] = distance[:, 0]
        covered[positions] = True
        processed += block_rows
        if row_group % 25 == 0 or row_group + 1 == parquet.num_row_groups:
            print(
                f"reconstructed oracle attribution {processed:,}/{EXPECTED_ROWS:,}",
                flush=True,
            )

    if processed != EXPECTED_ROWS or not covered.all():
        raise ValueError("candidate score coverage failed")
    hard_rmse = rmse(base["target"], selected_tvt)
    if abs(hard_rmse - EXPECTED_HARD_PRIMARY_RMSE) > 2e-6:
        raise ValueError(f"selected hard RMSE changed: {hard_rmse}")
    actual_from_tvt = np.abs(
        selected_tvt.astype(np.float64) - base["target"].astype(np.float64)
    )
    if float(np.max(np.abs(actual_from_tvt - selected_abs_error))) > 0.002:
        raise ValueError("actual_abs_error differs from candidate TVT and target")
    return {
        "selected_code": selected_code,
        "oracle_code": oracle_code,
        "selected_tvt": selected_tvt,
        "oracle_tvt": oracle_tvt,
        "selected_abs_error": selected_abs_error,
        "oracle_abs_error": oracle_abs_error,
        "md_since": md_since,
    }


def build_row_terms(
    base: dict[str, Any], surface: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    target = base["target"]
    exp274_sse = np.square(base["pred274"] - target)
    final_sse = np.square(base["pred264"] - target)
    selected_sse = np.square(surface["selected_tvt"] - target)
    oracle_sse = np.square(surface["oracle_tvt"] - target)
    terms = {
        "exp274_sse": exp274_sse,
        "final_sse": final_sse,
        "selected_sse": selected_sse,
        "oracle_sse": oracle_sse,
        "selection_regret_sse": selected_sse - oracle_sse,
    }
    if float(np.min(terms["selection_regret_sse"])) < -1e-6:
        raise ValueError("oracle selection regret is negative")
    terms["selector_code_match"] = (
        surface["selected_code"] == surface["oracle_code"]
    )
    terms["selector_tie_aware_correct"] = (
        surface["selected_abs_error"]
        <= surface["oracle_abs_error"] + TIE_TOLERANCE_FT
    )
    terms["selection_abs_regret"] = (
        surface["selected_abs_error"] - surface["oracle_abs_error"]
    )
    return terms


def summarize_mask(
    scope: str,
    mask: np.ndarray,
    base: dict[str, Any],
    surface: dict[str, np.ndarray],
    terms: dict[str, np.ndarray],
) -> dict[str, Any]:
    rows = int(mask.sum())
    if rows == 0:
        raise ValueError(f"empty scope: {scope}")
    target = base["target"][mask]
    exp274_sse = float(terms["exp274_sse"][mask].sum(dtype=np.float64))
    final_sse = float(terms["final_sse"][mask].sum(dtype=np.float64))
    selected_sse = float(terms["selected_sse"][mask].sum(dtype=np.float64))
    oracle_sse = float(terms["oracle_sse"][mask].sum(dtype=np.float64))
    total_delta = final_sse - exp274_sse
    oracle_term = oracle_sse - exp274_sse
    selection_term = selected_sse - oracle_sse
    stage_d_term = final_sse - selected_sse
    identity_error = oracle_term + selection_term + stage_d_term - total_delta
    return {
        "scope": scope,
        "rows": rows,
        "wells": int(np.unique(base["well_code"][mask]).size),
        "exp274_rmse": rmse(target, base["pred274"][mask]),
        "oracle_primary_rmse": rmse(target, surface["oracle_tvt"][mask]),
        "selected_hard_rmse": rmse(target, surface["selected_tvt"][mask]),
        "stage_d_final_rmse": rmse(target, base["pred264"][mask]),
        "selector_oracle_code_match_rate": float(
            terms["selector_code_match"][mask].mean()
        ),
        "selector_tie_aware_correct_rate": float(
            terms["selector_tie_aware_correct"][mask].mean()
        ),
        "selection_abs_regret_mean": float(
            terms["selection_abs_regret"][mask].mean()
        ),
        "oracle_vs_exp274_mse": oracle_term / rows,
        "selection_regret_mse": selection_term / rows,
        "selected_vs_exp274_mse": (selected_sse - exp274_sse) / rows,
        "stage_d_vs_selected_mse": stage_d_term / rows,
        "final_vs_exp274_mse": total_delta / rows,
        "oracle_vs_exp274_sse": oracle_term,
        "selection_regret_sse": selection_term,
        "stage_d_vs_selected_sse": stage_d_term,
        "final_vs_exp274_sse": total_delta,
        "attribution_identity_error": identity_error,
    }


def build_scope_summary(
    base: dict[str, Any],
    well: pd.DataFrame,
    surface: dict[str, np.ndarray],
    terms: dict[str, np.ndarray],
) -> pd.DataFrame:
    severe_by_well = well["delta_exp264_vs_exp274"].to_numpy(float) > 3.0
    severe_rows = severe_by_well[base["well_code"]]
    scopes = {
        "all": np.ones(EXPECTED_ROWS, bool),
        "worse_gt3": severe_rows,
        "other": ~severe_rows,
    }
    return pd.DataFrame(
        [
            summarize_mask(name, mask, base, surface, terms)
            for name, mask in scopes.items()
        ]
    )


def build_distance_summary(
    base: dict[str, Any],
    well: pd.DataFrame,
    surface: dict[str, np.ndarray],
    terms: dict[str, np.ndarray],
) -> pd.DataFrame:
    severe_by_well = well["delta_exp264_vs_exp274"].to_numpy(float) > 3.0
    severe_rows = severe_by_well[base["well_code"]]
    scopes = {
        "all": np.ones(EXPECTED_ROWS, bool),
        "worse_gt3": severe_rows,
        "other": ~severe_rows,
    }
    distance_code = pd.cut(
        surface["md_since"],
        bins=DISTANCE_BINS,
        labels=False,
        right=False,
    )
    records = []
    for scope, scope_mask in scopes.items():
        for code, label in enumerate(DISTANCE_LABELS):
            mask = scope_mask & (distance_code == code)
            record = summarize_mask(
                f"{scope}:{label}", mask, base, surface, terms
            )
            record["scope"] = scope
            record["distance_bucket"] = label
            records.append(record)
    return pd.DataFrame(records)


def build_by_well(
    base: dict[str, Any],
    well: pd.DataFrame,
    surface: dict[str, np.ndarray],
    terms: dict[str, np.ndarray],
) -> pd.DataFrame:
    codes = base["well_code"]
    counts = np.bincount(codes, minlength=EXPECTED_WELLS).astype(np.float64)

    def sum_by_well(values: np.ndarray) -> np.ndarray:
        return np.bincount(codes, weights=values, minlength=EXPECTED_WELLS)

    result = pd.DataFrame({"well": base["well_names"], "rows": counts.astype(int)})
    for name in [
        "exp274_sse",
        "oracle_sse",
        "selected_sse",
        "final_sse",
    ]:
        result[name] = sum_by_well(terms[name])
    result["oracle_vs_exp274_sse"] = (
        result["oracle_sse"] - result["exp274_sse"]
    )
    result["selection_regret_sse"] = (
        result["selected_sse"] - result["oracle_sse"]
    )
    result["selected_vs_exp274_sse"] = (
        result["selected_sse"] - result["exp274_sse"]
    )
    result["stage_d_vs_selected_sse"] = (
        result["final_sse"] - result["selected_sse"]
    )
    result["final_vs_exp274_sse"] = (
        result["final_sse"] - result["exp274_sse"]
    )
    result["exp274_rmse"] = np.sqrt(result["exp274_sse"] / counts)
    result["oracle_primary_rmse"] = np.sqrt(result["oracle_sse"] / counts)
    result["selected_hard_rmse"] = np.sqrt(result["selected_sse"] / counts)
    result["stage_d_final_rmse"] = np.sqrt(result["final_sse"] / counts)
    result["selector_oracle_code_match_rate"] = (
        sum_by_well(terms["selector_code_match"].astype(float)) / counts
    )
    result["selector_tie_aware_correct_rate"] = (
        sum_by_well(terms["selector_tie_aware_correct"].astype(float)) / counts
    )
    result["selection_abs_regret_mean"] = (
        sum_by_well(terms["selection_abs_regret"]) / counts
    )
    for name in [
        "oracle_vs_exp274",
        "selection_regret",
        "selected_vs_exp274",
        "stage_d_vs_selected",
        "final_vs_exp274",
    ]:
        result[f"{name}_mse"] = result[f"{name}_sse"] / counts
    result["attribution_identity_error"] = (
        result["oracle_vs_exp274_sse"]
        + result["selection_regret_sse"]
        + result["stage_d_vs_selected_sse"]
        - result["final_vs_exp274_sse"]
    )
    result["selected_hard_worse_than_exp274"] = (
        result["selected_vs_exp274_mse"] > 0
    )
    result["stage_d_worsens_selected_hard"] = (
        result["stage_d_vs_selected_mse"] > 0
    )
    result["mechanism"] = np.select(
        [
            result["selected_hard_worse_than_exp274"]
            & result["stage_d_worsens_selected_hard"],
            result["selected_hard_worse_than_exp274"]
            & ~result["stage_d_worsens_selected_hard"],
            ~result["selected_hard_worse_than_exp274"]
            & result["stage_d_worsens_selected_hard"],
        ],
        [
            "selector_failure_and_stage_d_worsens",
            "selector_failure_stage_d_mitigates",
            "stage_d_only_failure",
        ],
        default="neither_component_worsens",
    )
    selected_counts = np.zeros((EXPECTED_WELLS, PRIMARY_COUNT), np.int64)
    oracle_counts = np.zeros((EXPECTED_WELLS, PRIMARY_COUNT), np.int64)
    np.add.at(selected_counts, (codes, surface["selected_code"]), 1)
    np.add.at(oracle_counts, (codes, surface["oracle_code"]), 1)
    result["selected_dominant_candidate"] = np.asarray(CANDIDATES)[
        np.argmax(selected_counts, axis=1)
    ]
    result["oracle_dominant_candidate"] = np.asarray(CANDIDATES)[
        np.argmax(oracle_counts, axis=1)
    ]
    reference = well[
        [
            "well",
            "delta_exp264_vs_exp274",
            "exp274_rmse",
            "exp264_rmse",
        ]
    ].rename(
        columns={
            "exp274_rmse": "reference_exp274_rmse",
            "exp264_rmse": "reference_exp264_rmse",
        }
    )
    result = reference.merge(result, on="well", validate="one_to_one")
    if float(result["attribution_identity_error"].abs().max()) > 1e-7:
        raise ValueError("by-well attribution identity failed")
    return result


def build_confusion(
    base: dict[str, Any],
    well: pd.DataFrame,
    surface: dict[str, np.ndarray],
    terms: dict[str, np.ndarray],
) -> pd.DataFrame:
    severe_by_well = well["delta_exp264_vs_exp274"].to_numpy(float) > 3.0
    severe_rows = severe_by_well[base["well_code"]]
    pair_code = (
        surface["selected_code"].astype(np.int16) * PRIMARY_COUNT
        + surface["oracle_code"].astype(np.int16)
    )
    records = []
    for scope, mask in {
        "all": np.ones(EXPECTED_ROWS, bool),
        "worse_gt3": severe_rows,
        "other": ~severe_rows,
    }.items():
        counts = np.bincount(pair_code[mask], minlength=PRIMARY_COUNT**2)
        regret = np.bincount(
            pair_code[mask],
            weights=terms["selection_regret_sse"][mask],
            minlength=PRIMARY_COUNT**2,
        )
        selected_sse = np.bincount(
            pair_code[mask],
            weights=terms["selected_sse"][mask],
            minlength=PRIMARY_COUNT**2,
        )
        oracle_sse = np.bincount(
            pair_code[mask],
            weights=terms["oracle_sse"][mask],
            minlength=PRIMARY_COUNT**2,
        )
        selected_abs_error = np.bincount(
            pair_code[mask],
            weights=surface["selected_abs_error"][mask],
            minlength=PRIMARY_COUNT**2,
        )
        oracle_abs_error = np.bincount(
            pair_code[mask],
            weights=surface["oracle_abs_error"][mask],
            minlength=PRIMARY_COUNT**2,
        )
        for selected in range(PRIMARY_COUNT):
            for oracle in range(PRIMARY_COUNT):
                pair = selected * PRIMARY_COUNT + oracle
                if counts[pair] == 0:
                    continue
                records.append(
                    {
                        "scope": scope,
                        "selected_candidate": CANDIDATES[selected],
                        "oracle_candidate": CANDIDATES[oracle],
                        "rows": int(counts[pair]),
                        "row_share": float(counts[pair] / mask.sum()),
                        "selected_mae": float(
                            selected_abs_error[pair] / counts[pair]
                        ),
                        "selected_rmse": float(
                            np.sqrt(selected_sse[pair] / counts[pair])
                        ),
                        "oracle_mae": float(
                            oracle_abs_error[pair] / counts[pair]
                        ),
                        "oracle_rmse": float(
                            np.sqrt(oracle_sse[pair] / counts[pair])
                        ),
                        "selection_regret_sse": float(regret[pair]),
                        "selection_regret_mse": float(regret[pair] / counts[pair]),
                    }
                )
    return pd.DataFrame(records).sort_values(
        ["scope", "selection_regret_sse"], ascending=[True, False]
    )


def main() -> None:
    actual_sha = sha256_file(SCORE_PATH)
    if actual_sha != EXPECTED_SCORE_SHA256:
        raise ValueError(f"Stage C score SHA {actual_sha} != {EXPECTED_SCORE_SHA256}")
    well = pd.read_csv(WELL_PATH)
    if len(well) != EXPECTED_WELLS:
        raise ValueError("well comparison coverage changed")
    base = load_base_arrays(well)
    surface = reconstruct_selected_and_oracle(base)
    terms = build_row_terms(base, surface)
    scope_summary = build_scope_summary(base, well, surface, terms)
    distance_summary = build_distance_summary(base, well, surface, terms)
    by_well = build_by_well(base, well, surface, terms)
    confusion = build_confusion(base, well, surface, terms)
    top_worse = by_well.sort_values(
        "delta_exp264_vs_exp274", ascending=False
    ).head(100)

    output_frames = {
        "selector_oracle_scope_summary.csv": scope_summary,
        "selector_oracle_distance_summary.csv": distance_summary,
        "selector_oracle_by_well.csv": by_well,
        "selector_oracle_top100_worse_wells.csv": top_worse,
        "selector_oracle_confusion.csv": confusion,
    }
    for filename, frame in output_frames.items():
        frame.to_csv(ARTIFACT_DIR / filename, index=False)

    def scope_record(name: str) -> dict[str, Any]:
        row = scope_summary[scope_summary["scope"].eq(name)].iloc[0]
        return {
            key: (value.item() if hasattr(value, "item") else value)
            for key, value in row.to_dict().items()
        }

    severe = by_well["delta_exp264_vs_exp274"] > 3.0
    selector_caused = severe & (by_well["selected_vs_exp274_mse"] > 0)
    stage_d_mitigated = severe & (by_well["stage_d_vs_selected_mse"] < 0)
    stage_d_caused = severe & (by_well["stage_d_vs_selected_mse"] > 0)

    def pair_record(scope: str, selected: str, oracle: str) -> dict[str, Any]:
        row = confusion[
            confusion["scope"].eq(scope)
            & confusion["selected_candidate"].eq(selected)
            & confusion["oracle_candidate"].eq(oracle)
        ].iloc[0]
        return {
            key: (value.item() if hasattr(value, "item") else value)
            for key, value in row.to_dict().items()
        }

    def beam_record(scope: str, scope_rows: int) -> dict[str, Any]:
        rows = confusion[
            confusion["scope"].eq(scope)
            & confusion["selected_candidate"].eq("beam_mean")
        ]
        selected_rows = int(rows["rows"].sum())
        wrong_rows = int(
            rows.loc[rows["oracle_candidate"].ne("beam_mean"), "rows"].sum()
        )
        return {
            "scope": scope,
            "scope_rows": scope_rows,
            "beam_selected_rows": selected_rows,
            "beam_wrong_rows": wrong_rows,
            "beam_selected_row_rate": selected_rows / scope_rows,
            "beam_wrong_row_rate": wrong_rows / scope_rows,
            "wrong_rate_within_beam_selection": wrong_rows / selected_rows,
        }

    severe_scope_rows = int(
        scope_summary.loc[scope_summary["scope"].eq("worse_gt3"), "rows"].iloc[0]
    )
    beam_all = beam_record("all", EXPECTED_ROWS)
    beam_severe = beam_record("worse_gt3", severe_scope_rows)
    beam_other = beam_record("other", EXPECTED_ROWS - severe_scope_rows)
    beam_wrong_rate_lift = (
        beam_severe["beam_wrong_row_rate"] / beam_other["beam_wrong_row_rate"]
    )
    beam_expected_severe = (
        beam_severe["scope_rows"] * beam_other["beam_wrong_row_rate"]
    )
    summary = {
        "status": "complete_diagnostic_only",
        "rows": EXPECTED_ROWS,
        "wells": EXPECTED_WELLS,
        "question": (
            "Are wells worse than exp274 because the Stage C selector ranks the "
            "wrong candidate, or because Stage D worsens the selected hard path?"
        ),
        "definition": {
            "selected_candidate": "minimum predicted pred_abs_error among primary 11",
            "oracle_candidate": "minimum actual_abs_error among primary 11",
            "identity": (
                "final-exp274 = oracle-exp274 + selected-oracle + final-selected"
            ),
            "oracle_is_deployable": False,
            "hard_top1_is_stage_d_final": False,
        },
        "scopes": {
            "all": scope_record("all"),
            "worse_gt3": scope_record("worse_gt3"),
            "other": scope_record("other"),
        },
        "worse_gt3_wells": {
            "count": int(severe.sum()),
            "selected_hard_worse_than_exp274": int(selector_caused.sum()),
            "stage_d_mitigates_selected_hard": int(stage_d_mitigated.sum()),
            "stage_d_worsens_selected_hard": int(stage_d_caused.sum()),
            "selected_hard_worse_and_stage_d_mitigates": int(
                (selector_caused & stage_d_mitigated).sum()
            ),
            "selected_hard_worse_and_stage_d_worsens": int(
                (selector_caused & stage_d_caused).sum()
            ),
        },
        "focused_misselection": {
            "selected_selfgr_likpf_oracle_k16": {
                "all": pair_record(
                    "all", "selfgr_hmm_a070__likpf_mean", "exp226_k16"
                ),
                "worse_gt3": pair_record(
                    "worse_gt3",
                    "selfgr_hmm_a070__likpf_mean",
                    "exp226_k16",
                ),
            },
            "beam_wrong_selection": {
                "all": beam_all,
                "worse_gt3": beam_severe,
                "other": beam_other,
                "worse_gt3_vs_other_wrong_row_rate_lift": beam_wrong_rate_lift,
                "worse_gt3_expected_wrong_rows_at_other_rate": beam_expected_severe,
                "worse_gt3_excess_wrong_rows": (
                    beam_severe["beam_wrong_rows"] - beam_expected_severe
                ),
            },
        },
        "input_sha256": {
            "stage_c_outer_valid_candidate_score": actual_sha,
            "well_comparison": sha256_file(WELL_PATH),
            "exp274_oof_raw_gzip": sha256_file(EXP274_OOF_PATH),
            "exp264_stage_d_viewer": sha256_file(EXP264_VIEWER_PATH),
        },
        "output_sha256": {
            filename: sha256_file(ARTIFACT_DIR / filename)
            for filename in output_frames
        },
        "non_use_contract": [
            "oracle candidate uses actual TVT and is diagnostic only",
            "selector regret measures ranking within the fixed primary candidate set",
            "Stage D final is not the selected hard candidate prediction",
            "do not approve oracle routing, candidate removal, or Stage D changes from this posthoc readout alone",
        ],
    }
    summary_path = ARTIFACT_DIR / "selector_oracle_attribution_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\nScope attribution")
    print(scope_summary.to_string(index=False))
    print("\nTop worse wells")
    print(
        top_worse[
            [
                "well",
                "delta_exp264_vs_exp274",
                "selector_tie_aware_correct_rate",
                "oracle_primary_rmse",
                "selected_hard_rmse",
                "stage_d_final_rmse",
                "exp274_rmse",
                "selection_regret_mse",
                "stage_d_vs_selected_mse",
                "final_vs_exp274_mse",
            ]
        ].head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()
