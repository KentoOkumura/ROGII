from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from run_blend_audit import (
    CHUNK_ROWS,
    N_EXPECTED,
    assign_group_folds,
)

ROOT = Path(__file__).resolve().parents[2]
WORK = Path("/tmp/candidate_path_blend_audit_work")
OUTPUT = ROOT / "studies/candidate_path_blend_audit/outputs"
HIDDEN = ROOT / (
    "experiments/exp237_hmm_exp226_candidate_selector_on_exp183/inputs/"
    "exp115_hidden_like_spatial_holdout_from_ppt_fold_assignments.csv"
)


@dataclass(frozen=True)
class Spec:
    name: str
    members: tuple[str, ...]
    fixed_weights: tuple[float, ...] | None
    fold_weights: tuple[tuple[float, ...], ...] | None
    kind: str
    deploy_status: str


def pair_fold_weights(a: str, b: str) -> tuple[tuple[float, float], ...]:
    pairs = pd.read_csv(OUTPUT / "pair_blends.csv")
    row = pairs[(pairs.candidate_a == a) & (pairs.candidate_b == b)]
    if row.empty:
        row = pairs[(pairs.candidate_a == b) & (pairs.candidate_b == a)]
        if row.empty:
            raise KeyError((a, b))
        alphas = json.loads(row.iloc[0].crossfit_fold_weights_b)
        return tuple((alpha, 1.0 - alpha) for alpha in alphas)
    alphas = json.loads(row.iloc[0].crossfit_fold_weights_b)
    return tuple((1.0 - alpha, alpha) for alpha in alphas)


def triple_fold_weights(members: tuple[str, str, str]) -> tuple[tuple[float, ...], ...]:
    table = pd.read_csv(OUTPUT / "crossfit_triple_blends.csv")
    row = table[table.members == "|".join(members)]
    if row.empty:
        raise KeyError(members)
    return tuple(tuple(values) for values in json.loads(row.iloc[0].crossfit_fold_weights))


def specs() -> list[Spec]:
    path_triple = ("hmm_lgb_exp148", "exp226_k16", "exact_hmm")
    raw_path_triple = ("exp226_k16", "likpf_mean", "exact_hmm")
    hard_window_triple = ("exp226_k16", "exp192_likpf", "exact_hmm")
    selfgr_triple = ("exp226_k16", "selfgr_hmm_a070", "likpf_mean")
    blends = [
        Spec(
            "fixed_likpf_exact_50_50",
            ("likpf_mean", "exact_hmm"),
            (0.5, 0.5),
            None,
            "fixed_equal_path",
            "all_components_have_rawtest_generation",
        ),
        Spec(
            "fixed_hmmlgb_exp226_50_50",
            ("hmm_lgb_exp148", "exp226_k16"),
            (0.5, 0.5),
            None,
            "fixed_equal_path",
            "all_components_have_rawtest_inference",
        ),
        Spec(
            "fixed_exp226_selfgr_50_50",
            ("exp226_k16", "selfgr_hmm_a070"),
            (0.5, 0.5),
            None,
            "fixed_equal_path",
            "all_components_have_rawtest_generation",
        ),
        Spec(
            "fixed_exp226_w500_50_50",
            raw_path_triple,
            (0.5, 0.25, 0.25),
            None,
            "fixed_equal_derived_path",
            "all_components_have_rawtest_generation",
        ),
        Spec(
            "fixed_hmmlgb_exp226_selfgr_equal3",
            ("hmm_lgb_exp148", "exp226_k16", "selfgr_hmm_a070"),
            (1 / 3, 1 / 3, 1 / 3),
            None,
            "fixed_equal_path",
            "components_have_existing_rawtest_generation_paths",
        ),
        Spec(
            "fixed_hmmlgb_exp226_exact_equal3",
            path_triple,
            (1 / 3, 1 / 3, 1 / 3),
            None,
            "fixed_equal_path",
            "all_components_have_rawtest_generation",
        ),
        Spec(
            "fixed_hmmlgb_exp226_exact_50_30_20",
            path_triple,
            (0.5, 0.3, 0.2),
            None,
            "fixed_rounded_convex_path",
            "all_components_have_rawtest_generation_but_weights_are_post_audit",
        ),
        Spec(
            "fixed_hmmlgb_hmmshrink_exp226_exact_equal4",
            ("hmm_lgb_exp148", "hmm_exp218_shrink_a050", "exp226_k16", "exact_hmm"),
            (0.25,) * 4,
            None,
            "fixed_equal_path",
            "exp240_component_has_no_rawtest_port",
        ),
        Spec(
            "fixed_hmmlgb_hmmres_exp226_exact_equal4",
            ("hmm_lgb_exp148", "hmm_exp218_residual_scale", "exp226_k16", "exact_hmm"),
            (0.25,) * 4,
            None,
            "fixed_equal_path",
            "exp234_component_has_no_rawtest_port",
        ),
        Spec(
            "fixed_hmmlgb_exp226_exact_selfgr_equal4",
            ("hmm_lgb_exp148", "exp226_k16", "exact_hmm", "selfgr_hmm_a070"),
            (0.25,) * 4,
            None,
            "fixed_equal_path",
            "components_have_existing_rawtest_generation_paths",
        ),
        Spec(
            "fixed_hmmlgb_exp226_exact_peer_equal4",
            ("hmm_lgb_exp148", "exp226_k16", "exact_hmm", "hmm_peer_atlas"),
            (0.25,) * 4,
            None,
            "fixed_equal_path",
            "peer_atlas_has_no_rawtest_port",
        ),
        Spec(
            "crossfit_hmmlgb_exp226",
            ("hmm_lgb_exp148", "exp226_k16"),
            None,
            pair_fold_weights("hmm_lgb_exp148", "exp226_k16"),
            "crossfit_convex_path",
            "components_deployable_but_final_full_oof_weight_needed",
        ),
        Spec(
            "crossfit_hmmlgb_exp226_exact",
            path_triple,
            None,
            triple_fold_weights(path_triple),
            "crossfit_convex_path",
            "components_deployable_but_final_full_oof_weights_needed",
        ),
        Spec(
            "crossfit_exp226_likpf_exact",
            raw_path_triple,
            None,
            triple_fold_weights(raw_path_triple),
            "crossfit_convex_path",
            "all_components_have_rawtest_generation_but_final_weights_needed",
        ),
        Spec(
            "crossfit_exp226_exp192likpf_exact",
            hard_window_triple,
            None,
            triple_fold_weights(hard_window_triple),
            "crossfit_convex_path",
            "exp192_component_is_train_cache_only",
        ),
        Spec(
            "crossfit_exp226_selfgr_likpf",
            selfgr_triple,
            None,
            triple_fold_weights(selfgr_triple),
            "crossfit_convex_path",
            "all_components_have_rawtest_generation_but_final_weights_needed",
        ),
        Spec(
            "fixed_hmmlgb_exp237row_50_50",
            ("hmm_lgb_exp148", "exp237_row_selector"),
            (0.5, 0.5),
            None,
            "fixed_equal_mixed",
            "exp237_rawtest_parity_failed",
        ),
        Spec(
            "fixed_exp193_exp237row_50_50",
            ("exp193_lgb_mean", "exp237_row_selector"),
            (0.5, 0.5),
            None,
            "fixed_equal_model_output",
            "exp237_rawtest_parity_failed",
        ),
        Spec(
            "fixed_exp251prob_exp238_50_50",
            ("exp251_probability_row", "exp238_addonly"),
            (0.5, 0.5),
            None,
            "fixed_equal_model_output",
            "exp251_guard_failed_no_inference",
        ),
        Spec(
            "crossfit_exp251prob_exp255",
            ("exp251_probability_row", "exp255_assertive"),
            None,
            pair_fold_weights("exp251_probability_row", "exp255_assertive"),
            "crossfit_convex_model_output",
            "both_outputs_guard_failed_no_inference",
        ),
    ]
    parent_names = list(dict.fromkeys(name for item in blends for name in item.members))
    parents = [Spec(name, (name,), (1.0,), None, "single", "reference") for name in parent_names]
    return parents + blends


def prediction(
    block: np.ndarray,
    member_indices: list[int],
    row_folds: np.ndarray,
    spec: Spec,
) -> np.ndarray:
    values = block[:, member_indices].astype(np.float64)
    if spec.fixed_weights is not None:
        return values @ np.asarray(spec.fixed_weights, dtype=np.float64)
    if spec.fold_weights is None:
        raise ValueError(spec.name)
    out = np.empty(len(block), dtype=np.float64)
    for fold, weights in enumerate(spec.fold_weights):
        mask = row_folds == fold
        out[mask] = values[mask] @ np.asarray(weights, dtype=np.float64)
    return out


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    catalog = json.loads((WORK / "candidate_catalog.json").read_text())
    names = [item["name"] for item in catalog]
    name_to_idx = {name: i for i, name in enumerate(names)}
    meta = json.loads((WORK / "base_meta.json").read_text())
    well_names = meta["well_names"]
    n_candidates = len(names)
    pred = np.memmap(
        WORK / "predictions.f32", mode="r", dtype=np.float32, shape=(N_EXPECTED, n_candidates)
    )
    truth = np.memmap(WORK / "truth.f32", mode="r", dtype=np.float32, shape=N_EXPECTED)
    md_since = np.memmap(WORK / "md_since.f32", mode="r", dtype=np.float32, shape=N_EXPECTED)
    well_code = np.memmap(WORK / "well_code.u16", mode="r", dtype=np.uint16, shape=N_EXPECTED)
    counts_by_code = np.bincount(np.asarray(well_code), minlength=len(well_names))
    fold_by_code = assign_group_folds(well_names, counts_by_code)

    hidden = pd.read_csv(HIDDEN, dtype={"well_id": str}).set_index("well_id")
    spatial_codes = np.asarray(
        [hidden.loc[well, "verification_like_spatial_role"] == "valid" for well in well_names]
    )
    typewell_codes = np.asarray(
        [
            hidden.loc[well, "verification_like_typewell_purged_role"] == "valid"
            for well in well_names
        ]
    )
    scope_names = [
        "overall",
        "000_050",
        "050_100",
        "100_250",
        "250_500",
        "500_1000",
        "1000_plus",
        "hidden_spatial",
        "hidden_typewell_purged",
    ]
    selected = specs()
    accum = {
        item.name: {
            "sse": np.zeros(len(scope_names)),
            "sae": np.zeros(len(scope_names)),
            "within10": np.zeros(len(scope_names), dtype=np.int64),
            "count": np.zeros(len(scope_names), dtype=np.int64),
            "fold_sse": np.zeros(5),
            "fold_count": np.zeros(5, dtype=np.int64),
            "well_sse": np.zeros(len(well_names)),
        }
        for item in selected
    }

    for start in range(0, N_EXPECTED, CHUNK_ROWS):
        end = min(start + CHUNK_ROWS, N_EXPECTED)
        block = np.asarray(pred[start:end])
        y = np.asarray(truth[start:end], dtype=np.float64)
        md = np.asarray(md_since[start:end], dtype=np.float64)
        codes = np.asarray(well_code[start:end], dtype=np.int64)
        folds = fold_by_code[codes]
        scope_masks = [
            np.ones(end - start, dtype=bool),
            md < 50,
            (md >= 50) & (md < 100),
            (md >= 100) & (md < 250),
            (md >= 250) & (md < 500),
            (md >= 500) & (md < 1000),
            md >= 1000,
            spatial_codes[codes],
            typewell_codes[codes],
        ]
        for item in selected:
            member_indices = [name_to_idx[name] for name in item.members]
            p = prediction(block, member_indices, folds, item)
            residual = p - y
            sq = residual * residual
            absolute = np.abs(residual)
            store = accum[item.name]
            for scope_index, mask in enumerate(scope_masks):
                store["sse"][scope_index] += float(sq[mask].sum())
                store["sae"][scope_index] += float(absolute[mask].sum())
                store["within10"][scope_index] += int((absolute[mask] <= 10).sum())
                store["count"][scope_index] += int(mask.sum())
            for fold in range(5):
                mask = folds == fold
                store["fold_sse"][fold] += float(sq[mask].sum())
                store["fold_count"][fold] += int(mask.sum())
            store["well_sse"] += np.bincount(codes, weights=sq, minlength=len(well_names))

    scope_rows = []
    fold_rows = []
    for item in selected:
        store = accum[item.name]
        for idx, scope in enumerate(scope_names):
            count = int(store["count"][idx])
            scope_rows.append(
                {
                    "prediction": item.name,
                    "kind": item.kind,
                    "deploy_status": item.deploy_status,
                    "scope": scope,
                    "rows": count,
                    "rmse": np.sqrt(store["sse"][idx] / count),
                    "mae": store["sae"][idx] / count,
                    "within10": store["within10"][idx] / count,
                }
            )
        for fold in range(5):
            fold_rows.append(
                {
                    "prediction": item.name,
                    "kind": item.kind,
                    "fold": fold,
                    "rows": int(store["fold_count"][fold]),
                    "rmse": np.sqrt(store["fold_sse"][fold] / store["fold_count"][fold]),
                }
            )
    scope_df = pd.DataFrame(scope_rows)
    fold_df = pd.DataFrame(fold_rows)
    scope_df.to_csv(OUTPUT / "blend_scope_metrics.csv", index=False)
    fold_df.to_csv(OUTPUT / "blend_fold_metrics.csv", index=False)

    overall = scope_df[scope_df.scope == "overall"].set_index("prediction").rmse
    risk_rows = []
    scope_delta_rows = []
    for item in selected:
        if item.kind == "single":
            continue
        reference = min(item.members, key=lambda name: overall[name])
        blend_well = np.sqrt(accum[item.name]["well_sse"] / counts_by_code)
        ref_well = np.sqrt(accum[reference]["well_sse"] / counts_by_code)
        delta = blend_well - ref_well
        worst = int(np.argmax(delta))
        best = int(np.argmin(delta))
        blend_fold = fold_df[fold_df.prediction == item.name].set_index("fold").rmse
        ref_fold = fold_df[fold_df.prediction == reference].set_index("fold").rmse
        risk_rows.append(
            {
                "prediction": item.name,
                "kind": item.kind,
                "deploy_status": item.deploy_status,
                "reference": reference,
                "overall_rmse": overall[item.name],
                "delta_vs_reference": overall[item.name] - overall[reference],
                "folds_improved": int((blend_fold < ref_fold).sum()),
                "wells_improved": int((delta < 0).sum()),
                "wells_worsened": int((delta > 0).sum()),
                "median_well_delta": float(np.median(delta)),
                "p90_well_delta": float(np.quantile(delta, 0.90)),
                "p95_well_delta": float(np.quantile(delta, 0.95)),
                "max_well_regression": float(delta[worst]),
                "max_regression_well": well_names[worst],
                "max_well_improvement": float(delta[best]),
                "max_improvement_well": well_names[best],
            }
        )
        blend_scopes = scope_df[scope_df.prediction == item.name].set_index("scope")
        member_scopes = scope_df[scope_df.prediction.isin(item.members)]
        for scope in scope_names:
            member_rows = member_scopes[member_scopes.scope == scope]
            best_scope_row = member_rows.loc[member_rows.rmse.idxmin()]
            scope_delta_rows.append(
                {
                    "prediction": item.name,
                    "scope": scope,
                    "rmse": blend_scopes.loc[scope, "rmse"],
                    "best_member_on_scope": best_scope_row.prediction,
                    "best_member_scope_rmse": best_scope_row.rmse,
                    "delta_vs_best_member_on_scope": blend_scopes.loc[scope, "rmse"]
                    - best_scope_row.rmse,
                }
            )
    pd.DataFrame(risk_rows).sort_values("overall_rmse").to_csv(
        OUTPUT / "blend_well_risk.csv", index=False
    )
    pd.DataFrame(scope_delta_rows).sort_values(["prediction", "scope"]).to_csv(
        OUTPUT / "blend_scope_deltas.csv", index=False
    )


if __name__ == "__main__":
    main()
