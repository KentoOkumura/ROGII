from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDY_DIR = Path(__file__).resolve().parent
OUTPUT_PREFIX = "corr_prune_sanity_readout_on_exp148"

CORR_AUDIT_DIR_CANDIDATES = [
    Path("/tmp/kaggle-output/exp148_feature_correlation_audit_v2"),
    STUDY_DIR / "outputs" / "exp148_feature_correlation_audit_v2",
]
EXP148_TRAIN_ARTIFACT_DIR_CANDIDATES = [
    Path("/tmp/kaggle-output/exp148_train_v1/artifacts"),
]
EXP148_INFERENCE_ARTIFACT_DIR_CANDIDATES = [
    Path("/tmp/kaggle-output/exp148_inference_v7/artifacts"),
    Path("/tmp/kaggle-output/exp148_cpu_runtime_inference_v1/artifacts"),
    Path("/tmp/kaggle-output/exp148_inference/artifacts"),
]

EXP148_NAME = "exp148_learned_likelihood_fulltrain_addonly_on_exp092"


@dataclass(frozen=True)
class Candidate:
    bucket: str
    feature: str
    keep_feature: str | None
    relation: str
    evidence_level: str
    recommended_action: str
    rationale: str
    source_group: str


EXACT_PRUNE_CANDIDATES = [
    Candidate(
        "exact_prune_17",
        "sc_trust",
        None,
        "constant",
        "high",
        "drop_in_exact_prune",
        "single-value public replay trust column in the exp148 train surface",
        "public_replay_constant",
    ),
    Candidate(
        "exact_prune_17",
        "ll_candidate_tvt_likpf_mean_minus_likpf_mean_tvt",
        None,
        "constant_zero",
        "high",
        "drop_in_exact_prune",
        "likPF candidate TVT minus itself is a constant zero column",
        "learned_likelihood_candidate_tvt_delta",
    ),
    Candidate(
        "exact_prune_17",
        "ll_candidate_tvt_beam_mean_minus_last_known_tvt",
        "beam_mean_d",
        "existing_delta_duplicate",
        "high",
        "drop_in_exact_prune",
        "exp145 candidate TVT is the original beam_mean TVT; exp148 re-emits its last-known delta",
        "learned_likelihood_candidate_tvt_delta",
    ),
    Candidate(
        "exact_prune_17",
        "ll_candidate_tvt_beam_mean_minus_likpf_mean_tvt",
        "uproj_diff_beam_mean_minus_likpf_mean",
        "existing_disagreement_duplicate",
        "high",
        "drop_in_exact_prune",
        "beam_mean minus likPF disagreement already exists in the exp092 U-projection block",
        "learned_likelihood_candidate_tvt_delta",
    ),
    Candidate(
        "exact_prune_17",
        "ll_candidate_tvt_hyb_minus_last_known_tvt",
        "hyb_d",
        "existing_delta_duplicate",
        "high",
        "drop_in_exact_prune",
        "exp145 candidate TVT is the original hybrid TVT; exp148 re-emits its last-known delta",
        "learned_likelihood_candidate_tvt_delta",
    ),
    Candidate(
        "exact_prune_17",
        "ll_candidate_tvt_likpf_mean_minus_last_known_tvt",
        "likpf_mean_d",
        "existing_delta_duplicate",
        "high",
        "drop_in_exact_prune",
        "exp145 candidate TVT is the original likPF mean TVT; exp148 re-emits its last-known delta",
        "learned_likelihood_candidate_tvt_delta",
    ),
    Candidate(
        "exact_prune_17",
        "ll_candidate_tvt_pf_ancc_minus_last_known_tvt",
        "pf_ancc_delta",
        "existing_delta_duplicate",
        "high",
        "drop_in_exact_prune",
        "exp145 candidate TVT is the original PF ANCC TVT; exp148 re-emits its last-known delta",
        "learned_likelihood_candidate_tvt_delta",
    ),
    Candidate(
        "exact_prune_17",
        "ll_candidate_tvt_pf_ancc_minus_likpf_mean_tvt",
        "uproj_diff_pf_ancc_minus_likpf_mean",
        "existing_disagreement_duplicate",
        "high",
        "drop_in_exact_prune",
        "PF ANCC minus likPF disagreement already exists in the exp092 U-projection block",
        "learned_likelihood_candidate_tvt_delta",
    ),
    Candidate(
        "exact_prune_17",
        "ll_candidate_tvt_sc_ens_minus_last_known_tvt",
        "sc_ens_d",
        "existing_delta_duplicate",
        "high",
        "drop_in_exact_prune",
        (
            "exp145 candidate TVT is the original scale-consensus TVT; "
            "exp148 re-emits its last-known delta"
        ),
        "learned_likelihood_candidate_tvt_delta",
    ),
    Candidate(
        "exact_prune_17",
        "tda0",
        "gr_vs_tw_anc",
        "near_exact_public_replay_duplicate",
        "high",
        "drop_in_exact_prune",
        "typewell GR residual at zero anchor offset is the same diagnostic as gr_vs_tw_anc",
        "public_replay_anchor_duplicate",
    ),
    Candidate(
        "exact_prune_17",
        "dense_bias",
        "dense_rmse",
        "near_exact_public_replay_duplicate",
        "high",
        "drop_in_exact_prune",
        "dense ANCC bias and RMSE are effectively identical on the fixed train surface",
        "public_replay_dense_duplicate",
    ),
    Candidate(
        "exact_prune_17",
        "uproj_beam_mean_resid",
        "uproj_beam_mean_corr",
        "sign_flip_duplicate",
        "high",
        "drop_in_exact_prune",
        "exp092 writes both source_u - polynomial and polynomial - source_u",
        "u_projection_corr_resid_sign_flip",
    ),
    Candidate(
        "exact_prune_17",
        "uproj_beam_med_resid",
        "uproj_beam_med_corr",
        "sign_flip_duplicate",
        "high",
        "drop_in_exact_prune",
        "exp092 writes both source_u - polynomial and polynomial - source_u",
        "u_projection_corr_resid_sign_flip",
    ),
    Candidate(
        "exact_prune_17",
        "uproj_diff_pf_ancc_minus_pf_z",
        "pf_vs_z",
        "existing_disagreement_duplicate",
        "high",
        "drop_in_exact_prune",
        "PF ANCC minus PF Z in U-space collapses to the existing base pf_vs_z feature",
        "u_projection_base_disagreement_duplicate",
    ),
    Candidate(
        "exact_prune_17",
        "uproj_likpf_mean_resid",
        "uproj_likpf_mean_corr",
        "sign_flip_duplicate",
        "high",
        "drop_in_exact_prune",
        "exp092 writes both source_u - polynomial and polynomial - source_u",
        "u_projection_corr_resid_sign_flip",
    ),
    Candidate(
        "exact_prune_17",
        "uproj_pf_ancc_resid",
        "uproj_pf_ancc_corr",
        "sign_flip_duplicate",
        "high",
        "drop_in_exact_prune",
        "exp092 writes both source_u - polynomial and polynomial - source_u",
        "u_projection_corr_resid_sign_flip",
    ),
    Candidate(
        "exact_prune_17",
        "uproj_pf_z_resid",
        "uproj_pf_z_corr",
        "sign_flip_duplicate",
        "high",
        "drop_in_exact_prune",
        "exp092 writes both source_u - polynomial and polynomial - source_u",
        "u_projection_corr_resid_sign_flip",
    ),
]

FORMATION_LAST50_CANDIDATES = [
    Candidate(
        "formation_last50_followup_12",
        f"{prefix}50_{formation}",
        f"{prefix}w_{formation}",
        "formation_weighted_vs_last50_near_duplicate",
        "medium",
        "defer_to_separate_ablation",
        "weighted and last50 prefix formation-bias features are near-duplicates, but not exact",
        "public_replay_formation_bias_near_duplicate",
    )
    for prefix in ("bw", "tvtF")
    for formation in ("ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA")
]

LEARNED_LIKELIHOOD_SLIM_REVIEW = [
    Candidate(
        "learned_likelihood_slim_review",
        feature,
        None,
        "high_corr_or_redundancy_review",
        "medium",
        "review_only_do_not_include_in_exact_prune",
        (
            "candidate disagreement or spread signal is correlated with existing candidate values; "
            "verify separately"
        ),
        "learned_likelihood_slim_review",
    )
    for feature in (
        "ll_candidate_tvt_sc_ens_minus_likpf_mean_tvt",
        "ll_candidate_tvt_hyb_minus_likpf_mean_tvt",
        "ll_candidate_tvt_std",
        "ll_candidate_tvt_range",
    )
]

U_PROJECTION_SLIM_REVIEW = [
    Candidate(
        "u_projection_slim_review",
        feature,
        None,
        "u_projection_family_slim_review",
        "medium",
        "review_only_do_not_include_in_exact_prune",
        (
            "non-exact U-projection uncertainty/spread feature; "
            "keep out of the first drop-only ablation"
        ),
        "u_projection_slim_review",
    )
    for feature in (
        "uproj_pf_ancc_abs_resid",
        "uproj_pf_ancc_resid_mad",
        "uproj_pf_z_abs_resid",
        "uproj_pf_z_resid_mad",
        "uproj_beam_mean_abs_resid",
        "uproj_beam_mean_resid_mad",
        "uproj_beam_med_abs_resid",
        "uproj_beam_med_resid_mad",
        "uproj_likpf_mean_abs_resid",
        "uproj_likpf_mean_resid_mad",
        "uproj_source_u_std",
        "uproj_source_u_range",
        "uproj_corr_std",
        "uproj_corr_range",
    )
]

CODE_PATTERNS = {
    "learned_likelihood_candidate_tvt_delta": [
        (
            REPO_ROOT
            / "experiments/exp145_learned_likelihood_rawtest_feature_generator_parity/"
            "learned_likelihood_rawtest_feature_generator_parity.py",
            "class CandidateSpec",
        ),
        (
            REPO_ROOT
            / "experiments/exp145_learned_likelihood_rawtest_feature_generator_parity/"
            "learned_likelihood_rawtest_feature_generator_parity.py",
            "candidate_values",
        ),
        (
            REPO_ROOT
            / "experiments/exp148_learned_likelihood_fulltrain_addonly_on_exp092/"
            "learned_likelihood_fulltrain_addonly_on_exp092.py",
            "minus_last_known_tvt",
        ),
        (
            REPO_ROOT
            / "experiments/exp148_learned_likelihood_fulltrain_addonly_on_exp092/"
            "learned_likelihood_fulltrain_addonly_on_exp092.py",
            "minus_likpf_mean_tvt",
        ),
    ],
    "u_projection_corr_resid_sign_flip": [
        (
            REPO_ROOT
            / "experiments/exp092_u_projection_correction_disagreement_fullrun/"
            "u_projection_correction_disagreement_fullrun.py",
            "(source_u - poly)",
        ),
        (
            REPO_ROOT
            / "experiments/exp092_u_projection_correction_disagreement_fullrun/"
            "u_projection_correction_disagreement_fullrun.py",
            "(poly - source_u)",
        ),
    ],
    "u_projection_base_disagreement_duplicate": [
        (
            REPO_ROOT
            / "experiments/exp092_u_projection_correction_disagreement_fullrun/"
            "u_projection_correction_disagreement_fullrun.py",
            'diff_col = f"uproj_diff_{left}_minus_{right}"',
        ),
    ],
    "public_replay_constant": [
        (
            REPO_ROOT
            / "experiments/exp072_exp063_full_replay_feature_cache/"
            "public_notebook_replay_audit.py",
            "sc_trust",
        ),
    ],
    "public_replay_anchor_duplicate": [
        (
            REPO_ROOT
            / "experiments/exp072_exp063_full_replay_feature_cache/"
            "public_notebook_replay_audit.py",
            "ANCH_OFFS",
        ),
        (
            REPO_ROOT
            / "experiments/exp072_exp063_full_replay_feature_cache/"
            "public_notebook_replay_audit.py",
            "gr_vs_tw_anc",
        ),
        (
            REPO_ROOT
            / "experiments/exp072_exp063_full_replay_feature_cache/"
            "public_notebook_replay_audit.py",
            'f"tda{int(o)}"',
        ),
    ],
    "public_replay_dense_duplicate": [
        (
            REPO_ROOT
            / "experiments/exp072_exp063_full_replay_feature_cache/"
            "public_notebook_replay_audit.py",
            "dense_rmse",
        ),
        (
            REPO_ROOT
            / "experiments/exp072_exp063_full_replay_feature_cache/"
            "public_notebook_replay_audit.py",
            "dense_bias",
        ),
    ],
    "public_replay_formation_bias_near_duplicate": [
        (
            REPO_ROOT
            / "experiments/exp072_exp063_full_replay_feature_cache/"
            "public_notebook_replay_audit.py",
            "def seg_b_well",
        ),
        (
            REPO_ROOT
            / "experiments/exp072_exp063_full_replay_feature_cache/"
            "public_notebook_replay_audit.py",
            "tvtF50_",
        ),
        (
            REPO_ROOT
            / "experiments/exp072_exp063_full_replay_feature_cache/"
            "public_notebook_replay_audit.py",
            "bw50_",
        ),
    ],
}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if pd.isna(value):
        return None
    return value


def _first_existing(paths: list[Path], label: str) -> Path:
    for path in paths:
        if path.exists():
            return path
    joined = "\n".join(str(path) for path in paths)
    raise FileNotFoundError(f"Could not find {label}. Checked:\n{joined}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _optional_json(path: Path) -> dict[str, Any]:
    if path.exists():
        return _read_json(path)
    return {}


def _find_pattern(path: Path, pattern: str) -> str | None:
    if not path.exists():
        return None
    for lineno, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        if pattern in line:
            rel = path.relative_to(REPO_ROOT)
            return f"{rel}:{lineno}"
    return None


def _code_reference_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_group, refs in CODE_PATTERNS.items():
        for path, pattern in refs:
            rows.append(
                {
                    "source_group": source_group,
                    "path": str(path.relative_to(REPO_ROOT)) if path.exists() else str(path),
                    "pattern": pattern,
                    "reference": _find_pattern(path, pattern),
                }
            )
    return pd.DataFrame(rows)


def _schema_diff(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_name: str,
    right_name: str,
) -> pd.DataFrame:
    left_features = set(left["feature"].astype(str))
    right_features = set(right["feature"].astype(str))
    rows = []
    for feature in sorted(left_features | right_features):
        rows.append(
            {
                "feature": feature,
                f"in_{left_name}": feature in left_features,
                f"in_{right_name}": feature in right_features,
                "status": (
                    "both"
                    if feature in left_features and feature in right_features
                    else f"{left_name}_only"
                    if feature in left_features
                    else f"{right_name}_only"
                ),
            }
        )
    return pd.DataFrame(rows)


def _load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    corr_dir = Path(args.corr_dir) if args.corr_dir else _first_existing(
        CORR_AUDIT_DIR_CANDIDATES, "exp148 feature correlation audit directory"
    )
    train_artifact_dir = (
        Path(args.exp148_train_artifact_dir)
        if args.exp148_train_artifact_dir
        else _first_existing(
            EXP148_TRAIN_ARTIFACT_DIR_CANDIDATES,
            "exp148 train artifact directory",
        )
    )
    inference_artifact_dir = (
        Path(args.exp148_inference_artifact_dir)
        if args.exp148_inference_artifact_dir
        else _first_existing(
            EXP148_INFERENCE_ARTIFACT_DIR_CANDIDATES, "exp148 inference artifact directory"
        )
    )

    return {
        "corr_dir": corr_dir,
        "train_artifact_dir": train_artifact_dir,
        "inference_artifact_dir": inference_artifact_dir,
        "feature_readout": pd.read_csv(
            corr_dir / "exp148_feature_correlation_audit_feature_readout.csv"
        ),
        "top_pairs": pd.read_csv(corr_dir / "exp148_feature_correlation_audit_top500_pairs.csv"),
        "components_0995": pd.read_csv(
            corr_dir / "exp148_feature_correlation_audit_components_abs_ge_0995.csv"
        ),
        "corr_summary": _read_json(corr_dir / "exp148_feature_correlation_audit_summary.json"),
        "train_schema": pd.read_csv(
            train_artifact_dir / f"{EXP148_NAME}_feature_schema.csv"
        ),
        "train_importance": pd.read_csv(
            train_artifact_dir / f"{EXP148_NAME}_feature_importance_mean.csv"
        ),
        "train_summary": _read_json(train_artifact_dir / f"{EXP148_NAME}_summary.json"),
        "inference_schema": pd.read_csv(
            inference_artifact_dir / f"{EXP148_NAME}_inference_feature_schema.csv"
        ),
        "inference_summary": _optional_json(
            inference_artifact_dir / f"{EXP148_NAME}_inference_summary.json"
        ),
    }


def _candidate_frame(
    candidates: list[Candidate],
    readout: pd.DataFrame,
    top_pairs: pd.DataFrame,
    code_refs: pd.DataFrame,
    train_schema: pd.DataFrame,
    inference_schema: pd.DataFrame,
) -> pd.DataFrame:
    readout_by_feature = readout.set_index("feature", drop=False).to_dict("index")
    train_features = set(train_schema["feature"].astype(str))
    inference_features = set(inference_schema["feature"].astype(str))
    code_ref_by_group = (
        code_refs.dropna(subset=["reference"])
        .groupby("source_group")["reference"]
        .apply(lambda values: ";".join(values.astype(str)))
        .to_dict()
    )

    pair_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in top_pairs.itertuples(index=False):
        a = str(row.feature_a)
        b = str(row.feature_b)
        pair_lookup[(a, b)] = row._asdict()
        pair_lookup[(b, a)] = row._asdict()

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        feature_meta = readout_by_feature.get(candidate.feature, {})
        keep_meta = readout_by_feature.get(candidate.keep_feature or "", {})
        pair = (
            pair_lookup.get((candidate.feature, candidate.keep_feature))
            if candidate.keep_feature
            else None
        )
        rows.append(
            {
                **candidate.__dict__,
                "present_in_train_schema": candidate.feature in train_features,
                "present_in_inference_schema": candidate.feature in inference_features,
                "keep_present_in_train_schema": (
                    candidate.keep_feature in train_features if candidate.keep_feature else None
                ),
                "keep_present_in_inference_schema": (
                    candidate.keep_feature in inference_features if candidate.keep_feature else None
                ),
                "feature_family": feature_meta.get("family"),
                "mean_importance": feature_meta.get("mean_importance"),
                "importance_rank": feature_meta.get("importance_rank"),
                "max_abs_corr": feature_meta.get("max_abs_corr"),
                "max_corr_partner": feature_meta.get("max_corr_partner"),
                "nunique": feature_meta.get("nunique"),
                "std": feature_meta.get("std"),
                "target_corr": feature_meta.get("target_corr"),
                "keep_mean_importance": keep_meta.get("mean_importance"),
                "pair_corr": pair.get("corr") if pair else None,
                "pair_abs_corr": pair.get("abs_corr") if pair else None,
                "code_references": code_ref_by_group.get(candidate.source_group),
            }
        )
    return pd.DataFrame(rows)


def _config_fragments(exact_features: list[str], formation_features: list[str]) -> dict[str, Any]:
    return {
        "model": {
            "feature_ablation": {
                "selected_variant": "drop_exact_replacements_17",
                "active_variants": [
                    {
                        "name": "drop_exact_replacements_17",
                        "description": (
                            "Drop only the high-confidence constant, exact duplicate, sign-flip, "
                            "and already-emitted delta/disagreement columns identified by "
                            "corr_prune_sanity_readout_on_exp148."
                        ),
                        "feature_groups": [
                            "projection_correction",
                            "u_disagreement",
                            "learned_likelihood_confidence",
                        ],
                        "drop_columns": exact_features,
                        "enabled": True,
                    },
                    {
                        "name": "drop_exact_plus_formation_last50_29",
                        "description": (
                            "Follow-up only: add formation last50 near-duplicate drops "
                            "to the exact "
                            "17-column prune list. Do not run together with the first exact-prune "
                            "ablation unless explicitly selected."
                        ),
                        "feature_groups": [
                            "projection_correction",
                            "u_disagreement",
                            "learned_likelihood_confidence",
                        ],
                        "drop_columns": [*exact_features, *formation_features],
                        "enabled": False,
                    },
                ],
            }
        },
        "runtime_note": {
            "control_retraining": False,
            "gpu_training_required_for_this_readout": False,
            "expected_exact_prune_feature_count": 294 - len(exact_features),
            "expected_exact_plus_formation_feature_count": 294
            - len([*exact_features, *formation_features]),
        },
    }


def _candidate_group_json(candidate_df: pd.DataFrame) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for bucket, frame in candidate_df.groupby("bucket", sort=False):
        groups[str(bucket)] = {
            "features": frame["feature"].tolist(),
            "recommended_actions": sorted(frame["recommended_action"].dropna().unique().tolist()),
            "evidence_levels": sorted(frame["evidence_level"].dropna().unique().tolist()),
        }
    return groups


def run(args: argparse.Namespace) -> dict[str, Any]:
    inputs = _load_inputs(args)
    output_dir = Path(args.output_dir) if args.output_dir else (
        STUDY_DIR / "outputs" / OUTPUT_PREFIX
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    code_refs = _code_reference_table()
    all_candidates = [
        *EXACT_PRUNE_CANDIDATES,
        *FORMATION_LAST50_CANDIDATES,
        *LEARNED_LIKELIHOOD_SLIM_REVIEW,
        *U_PROJECTION_SLIM_REVIEW,
    ]
    candidate_df = _candidate_frame(
        all_candidates,
        inputs["feature_readout"],
        inputs["top_pairs"],
        code_refs,
        inputs["train_schema"],
        inputs["inference_schema"],
    )

    exact_features = [candidate.feature for candidate in EXACT_PRUNE_CANDIDATES]
    formation_features = [candidate.feature for candidate in FORMATION_LAST50_CANDIDATES]
    config_fragment = _config_fragments(exact_features, formation_features)

    constants = inputs["feature_readout"].loc[
        inputs["feature_readout"]["nunique"].le(1)
    ].sort_values("feature")
    train_inference_diff = _schema_diff(
        inputs["train_schema"], inputs["inference_schema"], "train", "inference"
    )

    exp145_train_schema = pd.read_csv(
        REPO_ROOT
        / "experiments/exp145_learned_likelihood_rawtest_feature_generator_parity/"
        "kaggle/output/train_v2/artifacts/"
        "exp145_learned_likelihood_rawtest_feature_generator_parity_feature_schema.csv"
    )
    exp145_rawtest_schema = pd.read_csv(
        REPO_ROOT
        / "experiments/exp145_learned_likelihood_rawtest_feature_generator_parity/"
        "kaggle/output/inference_v3/artifacts/"
        "exp145_learned_likelihood_rawtest_feature_generator_parity_feature_schema.csv"
    )
    exp145_schema_diff = _schema_diff(
        exp145_train_schema, exp145_rawtest_schema, "exp145_train", "exp145_rawtest"
    )

    candidate_df.to_csv(output_dir / f"{OUTPUT_PREFIX}_drop_candidates.csv", index=False)
    constants.to_csv(output_dir / f"{OUTPUT_PREFIX}_constant_features.csv", index=False)
    inputs["components_0995"].to_csv(
        output_dir / f"{OUTPUT_PREFIX}_components_abs_ge_0995.csv", index=False
    )
    code_refs.to_csv(output_dir / f"{OUTPUT_PREFIX}_code_references.csv", index=False)
    train_inference_diff.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_exp148_train_inference_schema_diff.csv", index=False
    )
    exp145_schema_diff.to_csv(
        output_dir / f"{OUTPUT_PREFIX}_exp145_train_rawtest_schema_diff.csv", index=False
    )
    (output_dir / f"{OUTPUT_PREFIX}_config_fragment.yaml").write_text(
        yaml.safe_dump(config_fragment, sort_keys=False)
    )
    (output_dir / f"{OUTPUT_PREFIX}_config_fragment.json").write_text(
        json.dumps(_jsonable(config_fragment), indent=2)
    )
    (output_dir / f"{OUTPUT_PREFIX}_candidate_groups.json").write_text(
        json.dumps(_jsonable(_candidate_group_json(candidate_df)), indent=2)
    )

    summary = {
        "study": OUTPUT_PREFIX,
        "status": "completed",
        "purpose": (
            "No-training sanity readout for exact/near feature prune candidates before "
            "exact_replacement_prune_on_exp148."
        ),
        "inputs": {
            "corr_dir": str(inputs["corr_dir"]),
            "train_artifact_dir": str(inputs["train_artifact_dir"]),
            "inference_artifact_dir": str(inputs["inference_artifact_dir"]),
        },
        "exp148_anchor": {
            "cv_lgb_mean": 8.50128118189582,
            "public_lb": 7.960,
            "feature_count": 294,
        },
        "corr_audit_summary": {
            "sample_rows_used": inputs["corr_summary"]["sample_policy"]["sample_rows_used"],
            "feature_counts": inputs["corr_summary"]["feature_counts"],
            "constant_or_single_value_features": inputs["corr_summary"][
                "constant_or_single_value_features"
            ],
            "pair_counts_by_abs_corr_threshold": inputs["corr_summary"][
                "pair_counts_by_abs_corr_threshold"
            ],
        },
        "candidate_counts": {
            "exact_prune_17": len(EXACT_PRUNE_CANDIDATES),
            "formation_last50_followup_12": len(FORMATION_LAST50_CANDIDATES),
            "learned_likelihood_slim_review": len(LEARNED_LIKELIHOOD_SLIM_REVIEW),
            "u_projection_slim_review": len(U_PROJECTION_SLIM_REVIEW),
        },
        "schema_parity": {
            "exp148_train_inference_diff_nonboth": int(
                train_inference_diff["status"].ne("both").sum()
            ),
            "exp145_train_rawtest_diff_nonboth": int(exp145_schema_diff["status"].ne("both").sum()),
        },
        "outputs": {
            "drop_candidates": f"{OUTPUT_PREFIX}_drop_candidates.csv",
            "constant_features": f"{OUTPUT_PREFIX}_constant_features.csv",
            "components_abs_ge_0995": f"{OUTPUT_PREFIX}_components_abs_ge_0995.csv",
            "code_references": f"{OUTPUT_PREFIX}_code_references.csv",
            "exp148_train_inference_schema_diff": (
                f"{OUTPUT_PREFIX}_exp148_train_inference_schema_diff.csv"
            ),
            "exp145_train_rawtest_schema_diff": (
                f"{OUTPUT_PREFIX}_exp145_train_rawtest_schema_diff.csv"
            ),
            "config_fragment_yaml": f"{OUTPUT_PREFIX}_config_fragment.yaml",
            "config_fragment_json": f"{OUTPUT_PREFIX}_config_fragment.json",
            "candidate_groups": f"{OUTPUT_PREFIX}_candidate_groups.json",
        },
    }
    (output_dir / f"{OUTPUT_PREFIX}_summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2)
    )
    print(json.dumps(_jsonable(summary), indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a no-training corr-prune sanity readout for exp148."
    )
    parser.add_argument(
        "--corr-dir",
        default=None,
        help="Directory with exp148 correlation audit CSVs",
    )
    parser.add_argument(
        "--exp148-train-artifact-dir",
        default=None,
        help="Directory with exp148 train artifact schema/importance/summary files",
    )
    parser.add_argument(
        "--exp148-inference-artifact-dir",
        default=None,
        help="Directory with exp148 inference artifact schema/summary files",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory for readout files")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
