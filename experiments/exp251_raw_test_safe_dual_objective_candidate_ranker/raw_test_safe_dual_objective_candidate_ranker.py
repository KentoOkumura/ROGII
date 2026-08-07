from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import candidate_ranker_engine as engine
import numpy as np
import pandas as pd
import rawtest_feature_builder as rawtest
from copcf_rawtest_regeneration import attach_regenerated_copcf
from settings import ExperimentPaths, get_nested, load_config

OUTPUT_PREFIX = "exp251_raw_test_safe_dual_objective_candidate_ranker"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n")


def _sample_sorted_rows(row_count: int, limit: int | None, *, seed: int) -> np.ndarray:
    indices = np.arange(row_count, dtype=np.int64)
    if limit is None or row_count <= int(limit):
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=int(limit), replace=False))


def _candidate_values(frame: pd.DataFrame, candidates: list[Any]) -> np.ndarray:
    values = np.column_stack(
        [
            pd.to_numeric(frame[item.column], errors="coerce").to_numpy(np.float32)
            for item in candidates
        ]
    )
    if not np.isfinite(values).all():
        bad = np.argwhere(~np.isfinite(values))[:10].tolist()
        raise ValueError(f"raw-test candidate matrix contains non-finite values: {bad}")
    return values


def _assemble_rawtest_candidate_surface(
    *,
    config: dict[str, Any],
    paths: ExperimentPaths,
    candidates: list[Any],
    train_reference_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    base, base_meta = rawtest._load_base_test_frame(config, paths)
    frame = rawtest._base_candidates(base)
    frame, hmm_meta = rawtest._attach_hmm_candidates(frame, config, paths)
    frame, exp226_meta = rawtest._attach_exp226_candidate(frame, config)
    frame, copcf_columns, copcf_meta = attach_regenerated_copcf(
        frame,
        config,
        paths,
        train_reference_frame=train_reference_frame,
    )
    frame["target"] = np.float32(0.0)
    frame["true_tvt"] = frame["last_known_tvt"].to_numpy(np.float32)
    values = _candidate_values(frame, candidates)
    return (
        frame,
        values,
        {
            "base": base_meta,
            "hmm": hmm_meta,
            "exp226": exp226_meta,
            "copcf": copcf_meta,
            "copcf_generated_base_feature_count": len(copcf_columns),
            "rows": int(len(frame)),
            "wells": int(frame["well"].nunique()),
        },
    )


def _rawtest_base_feature_columns(
    frame: pd.DataFrame,
    candidate_values: np.ndarray,
    candidates: list[Any],
    train_base_features: list[str],
    config: dict[str, Any],
) -> list[str]:
    prepared, engineered, values_check, _ = engine.parent.add_candidate_labels_and_features(
        frame, candidates, include_candidate_values=False
    )
    if not np.array_equal(values_check, candidate_values):
        raise ValueError("raw-test candidate values changed during feature preflight")
    regenerated = set(engineered)
    regenerated.update(
        column
        for column in train_base_features
        if column.startswith("multiobs_") or column.startswith("likpf_multiobs_")
    )
    available = set(prepared.columns).union(regenerated)
    return [column for column in train_base_features if column in available]


def _finite_summary(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    if column not in frame.columns:
        return {
            "rows": int(len(frame)),
            "finite_rows": 0,
            "missing_rate": 1.0,
            "mean": np.nan,
            "std": np.nan,
            "q01": np.nan,
            "q50": np.nan,
            "q99": np.nan,
            "unique": 0,
        }
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(np.float64)
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {
            "rows": int(len(values)),
            "finite_rows": 0,
            "missing_rate": 1.0,
            "mean": np.nan,
            "std": np.nan,
            "q01": np.nan,
            "q50": np.nan,
            "q99": np.nan,
            "unique": 0,
        }
    return {
        "rows": int(len(values)),
        "finite_rows": int(len(finite)),
        "missing_rate": float(1.0 - len(finite) / max(len(values), 1)),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "q01": float(np.quantile(finite, 0.01)),
        "q50": float(np.quantile(finite, 0.50)),
        "q99": float(np.quantile(finite, 0.99)),
        "unique": int(pd.Series(finite).nunique(dropna=True)),
    }


def _distribution_shift(
    train: pd.DataFrame, rawtest_frame: pd.DataFrame, column: str
) -> tuple[float, float]:
    if column not in rawtest_frame.columns:
        return np.nan, np.nan
    train_values = pd.to_numeric(train[column], errors="coerce").to_numpy(np.float64)
    test_values = pd.to_numeric(rawtest_frame[column], errors="coerce").to_numpy(np.float64)
    train_values = train_values[np.isfinite(train_values)]
    test_values = test_values[np.isfinite(test_values)]
    if not len(train_values) or not len(test_values):
        return np.nan, np.nan
    scale = max(float(np.std(train_values)), 1e-6)
    standardized_mean_diff = float((np.mean(test_values) - np.mean(train_values)) / scale)
    edges = np.unique(np.quantile(train_values, np.linspace(0.0, 1.0, 11)))
    if len(edges) < 3:
        return standardized_mean_diff, 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    train_hist = np.histogram(train_values, bins=edges)[0].astype(np.float64)
    test_hist = np.histogram(test_values, bins=edges)[0].astype(np.float64)
    train_share = np.maximum(train_hist / max(train_hist.sum(), 1.0), 1e-6)
    test_share = np.maximum(test_hist / max(test_hist.sum(), 1.0), 1e-6)
    psi = float(np.sum((test_share - train_share) * np.log(test_share / train_share)))
    return standardized_mean_diff, psi


def _provenance(column: str, generated_on_rawtest: bool, config: dict[str, Any]) -> str:
    disallowed_prefixes = tuple(
        str(value) for value in get_nested(config, "feature_audit.disallowed_prefixes") or []
    )
    disallowed_exact = {
        str(value) for value in get_nested(config, "feature_audit.disallowed_exact_columns") or []
    }
    regenerated_prefixes = tuple(
        str(value) for value in get_nested(config, "feature_audit.regenerated_prefixes") or []
    )
    if column in disallowed_exact:
        return "train_only_auxiliary"
    if not generated_on_rawtest:
        return "not_generated_on_raw_test"
    if disallowed_prefixes and column.startswith(disallowed_prefixes):
        return "disallowed_generated_prefix"
    if regenerated_prefixes and column.startswith(regenerated_prefixes):
        return "raw_test_full_train_prior_or_target_free_cluster"
    if column.startswith(("multiobs_", "candidate_multiobs_", "likpf_multiobs_")):
        return "raw_test_multiobs_regeneration"
    if column.startswith(("hmm_", "self_gr_", "blend_likpf_hmm_")):
        return "raw_test_hmm_regeneration"
    if column.startswith("view_") or column in {
        "candidate_abs_minus_view_mean",
        "candidate_z_within_view",
        "candidate_score_gap_from_view_best",
    }:
        return "raw_test_candidate_set_context"
    candidate_tokens = (
        "candidate_",
        "_minus_last",
        "_vs_",
        "candidate_mean",
        "candidate_std",
        "candidate_range",
    )
    if column.startswith("candidate_") or any(token in column for token in candidate_tokens[1:]):
        return "raw_test_candidate_derivation"
    return "raw_test_base_cache"


def _audit_features(
    train_long: pd.DataFrame,
    rawtest_long: pd.DataFrame,
    train_schema: list[str],
    rawtest_schema: list[str],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    max_missing = float(get_nested(config, "feature_audit.max_missing_rate") or 0.0)
    max_missing_delta = float(get_nested(config, "feature_audit.max_missing_rate_delta") or 0.0)
    smd_warning = float(
        get_nested(config, "feature_audit.distribution_warning_abs_standardized_mean_diff")
        or np.inf
    )
    psi_warning = float(get_nested(config, "feature_audit.distribution_warning_psi") or np.inf)
    allowed_provenance = {
        str(value) for value in get_nested(config, "feature_audit.allowed_provenance") or []
    }
    rawtest_columns = set(rawtest_schema)
    rows: list[dict[str, Any]] = []
    selected: list[str] = []
    for order, column in enumerate(train_schema):
        generated = column in rawtest_columns
        provenance = _provenance(column, generated, config)
        train_stats = _finite_summary(train_long, column)
        test_stats = _finite_summary(rawtest_long, column)
        smd, psi = _distribution_shift(train_long, rawtest_long, column)
        fallback_delta = abs(test_stats["missing_rate"] - train_stats["missing_rate"])
        reasons: list[str] = []
        if not generated:
            reasons.append("not_generated_on_raw_test")
        if provenance not in allowed_provenance:
            reasons.append(f"provenance:{provenance}")
        if train_stats["missing_rate"] > max_missing:
            reasons.append("train_fallback_rate")
        if test_stats["missing_rate"] > max_missing:
            reasons.append("rawtest_fallback_rate")
        if fallback_delta > max_missing_delta:
            reasons.append("fallback_rate_delta")
        passed = not reasons
        if passed:
            selected.append(column)
        distribution_warning = bool(
            (np.isfinite(smd) and abs(smd) > smd_warning)
            or (np.isfinite(psi) and psi > psi_warning)
        )
        rows.append(
            {
                "feature_order": order,
                "feature": column,
                "provenance": provenance,
                "generated_on_rawtest": generated,
                "train_missing_rate": train_stats["missing_rate"],
                "rawtest_missing_rate": test_stats["missing_rate"],
                "missing_rate_delta": fallback_delta,
                "train_mean": train_stats["mean"],
                "rawtest_mean": test_stats["mean"],
                "train_std": train_stats["std"],
                "rawtest_std": test_stats["std"],
                "train_q01": train_stats["q01"],
                "rawtest_q01": test_stats["q01"],
                "train_q50": train_stats["q50"],
                "rawtest_q50": test_stats["q50"],
                "train_q99": train_stats["q99"],
                "rawtest_q99": test_stats["q99"],
                "standardized_mean_diff": smd,
                "psi": psi,
                "distribution_warning": distribution_warning,
                "selected": passed,
                "exclusion_reasons": "|".join(reasons),
            }
        )
    audit = pd.DataFrame(rows)
    expected = int(get_nested(config, "feature_audit.expected_parent_long_feature_count") or 0)
    expected_selected = int(
        get_nested(config, "feature_audit.expected_selected_long_feature_count") or 0
    )
    expected_regenerated = int(
        get_nested(config, "feature_audit.expected_regenerated_prefix_feature_count") or 0
    )
    minimum = int(get_nested(config, "feature_audit.min_selected_long_feature_count") or 1)
    regenerated_prefixes = tuple(
        str(value)
        for value in get_nested(config, "feature_audit.regenerated_prefixes") or []
    )
    regenerated_mask = audit["feature"].str.startswith(regenerated_prefixes)
    regenerated = audit.loc[regenerated_mask]
    decision = {
        "pass": bool(
            len(train_schema) == expected
            and len(selected) >= minimum
            and (expected_selected <= 0 or len(selected) == expected_selected)
        ),
        "checks": {
            "parent_feature_count_exact": len(train_schema) == expected,
            "minimum_selected_features": len(selected) >= minimum,
            "selected_feature_count_exact": (
                expected_selected <= 0 or len(selected) == expected_selected
            ),
            "regenerated_feature_count_exact": (
                expected_regenerated <= 0 or len(regenerated) == expected_regenerated
            ),
            "all_regenerated_features_generated_on_rawtest": bool(
                regenerated["generated_on_rawtest"].all()
            ),
            "all_regenerated_features_selected": bool(regenerated["selected"].all()),
            "selected_features_generated_on_rawtest": bool(
                audit.loc[audit["selected"], "generated_on_rawtest"].all()
            ),
            "no_disallowed_selected_provenance": bool(
                audit.loc[audit["selected"], "provenance"].isin(allowed_provenance).all()
            ),
        },
        "parent_feature_count": len(train_schema),
        "rawtest_generated_feature_count": int(audit["generated_on_rawtest"].sum()),
        "regenerated_feature_count": int(len(regenerated)),
        "selected_regenerated_feature_count": int(regenerated["selected"].sum()),
        "selected_feature_count": len(selected),
        "excluded_feature_count": int((~audit["selected"]).sum()),
        "distribution_warning_count": int(audit["distribution_warning"].sum()),
    }
    decision["pass"] = bool(decision["pass"] and all(decision["checks"].values()))
    return audit, selected, decision


def synthetic_feature_audit_contract_test() -> dict[str, Any]:
    config = load_config()
    config = dict(config)
    config["feature_audit"] = dict(config["feature_audit"])
    config["feature_audit"]["expected_parent_long_feature_count"] = 5
    config["feature_audit"]["expected_selected_long_feature_count"] = 3
    config["feature_audit"]["expected_regenerated_prefix_feature_count"] = 1
    config["feature_audit"]["min_selected_long_feature_count"] = 2
    train = pd.DataFrame(
        {
            "last_known_tvt": [100.0, 101.0, 102.0, 103.0],
            "candidate_minus_last": [1.0, 2.0, 3.0, 4.0],
            "copcf_regenerated": [0.0, 1.0, 0.0, 1.0],
            "exp226_gr_delta": [2.0, 2.5, 3.0, 3.5],
            "train_only_context": [7.0, 8.0, 9.0, 10.0],
        }
    )
    raw = pd.DataFrame(
        {
            "last_known_tvt": [110.0, 111.0, 112.0, 113.0],
            "candidate_minus_last": [1.5, 2.5, 3.5, 4.5],
            "copcf_regenerated": [0.0, 0.0, 0.0, 0.0],
            "exp226_gr_delta": [4.0, 4.0, 4.0, 4.0],
        }
    )
    train_schema = train.columns.tolist()
    raw_schema = raw.columns.tolist()
    first = _audit_features(train, raw, train_schema, raw_schema, config)
    second = _audit_features(train, raw, train_schema, raw_schema, config)
    selected = first[1]
    expected_selected = ["last_known_tvt", "candidate_minus_last", "copcf_regenerated"]
    if selected != expected_selected:
        raise AssertionError(f"unexpected synthetic selected schema: {selected}")
    if not first[2]["pass"]:
        raise AssertionError(f"synthetic feature audit should pass: {first[2]}")
    deterministic = first[0].equals(second[0]) and first[1:] == second[1:]
    if not deterministic:
        raise AssertionError("synthetic feature audit is not deterministic")
    return {
        "deterministic": True,
        "parent_feature_count": len(train_schema),
        "selected_features": selected,
        "excluded_features": first[0].loc[~first[0]["selected"], "feature"].tolist(),
    }


def run_feature_audit(
    *,
    output_dir: str | Path,
    cache_path: str | Path | None = None,
    schema_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config()
    paths = ExperimentPaths()
    (
        train_frame,
        candidates,
        train_candidate_values,
        _train_oracle,
        train_base_features,
        train_source_meta,
    ) = engine.assemble_parent_candidate_surface(
        cache_path=cache_path,
        schema_path=schema_path,
        max_rows=None,
    )
    sample_rows = _sample_sorted_rows(
        len(train_frame),
        get_nested(config, "feature_audit.max_train_base_rows"),
        seed=engine.stable_seed(OUTPUT_PREFIX, "feature_audit_train_rows"),
    )
    train_sample = train_frame.iloc[sample_rows]
    train_long, _, _, train_schema = engine.build_candidate_long_view(
        train_sample,
        train_candidate_values[sample_rows],
        np.ones((len(sample_rows), len(candidates)), dtype=bool),
        candidates=candidates,
        base_feature_columns=train_base_features,
        config=config,
        raw_cache={},
        horizontal_dir=paths.train_data_dir,
    )
    raw_frame, raw_candidate_values, raw_source_meta = _assemble_rawtest_candidate_surface(
        config=config,
        paths=paths,
        candidates=candidates,
        train_reference_frame=train_frame,
    )
    raw_limit = get_nested(config, "feature_audit.max_rawtest_base_rows")
    raw_rows = _sample_sorted_rows(
        len(raw_frame),
        int(raw_limit) if raw_limit is not None else None,
        seed=engine.stable_seed(OUTPUT_PREFIX, "feature_audit_rawtest_rows"),
    )
    raw_sample = raw_frame.iloc[raw_rows]
    raw_base_features = _rawtest_base_feature_columns(
        raw_sample,
        raw_candidate_values[raw_rows],
        candidates,
        train_base_features,
        config,
    )
    rawtest_long, _, _, rawtest_schema = engine.build_candidate_long_view(
        raw_sample,
        raw_candidate_values[raw_rows],
        np.ones((len(raw_rows), len(candidates)), dtype=bool),
        candidates=candidates,
        base_feature_columns=raw_base_features,
        config=config,
        raw_cache={},
        horizontal_dir=paths.test_data_dir,
    )
    audit, selected_features, decision = _audit_features(
        train_long, rawtest_long, train_schema, rawtest_schema, config
    )
    audit_path = output_dir / f"{OUTPUT_PREFIX}_feature_audit.csv"
    schema_path_out = output_dir / f"{OUTPUT_PREFIX}_selected_feature_schema.csv"
    contract_path = output_dir / f"{OUTPUT_PREFIX}_feature_contract.json"
    train_sample_path = output_dir / f"{OUTPUT_PREFIX}_train_feature_sample.csv.gz"
    rawtest_sample_path = output_dir / f"{OUTPUT_PREFIX}_rawtest_feature_sample.csv.gz"
    summary_path = output_dir / f"{OUTPUT_PREFIX}_feature_audit_summary.json"
    audit.to_csv(audit_path, index=False)
    pd.DataFrame(
        {"feature_order": np.arange(len(selected_features)), "feature": selected_features}
    ).to_csv(schema_path_out, index=False)
    persist_limit = int(get_nested(config, "feature_audit.persist_max_long_rows") or 50000)
    sample_key_columns = ["id", "well", "candidate_index"]
    train_long.loc[:, [*sample_key_columns, *selected_features]].head(persist_limit).to_csv(
        train_sample_path, index=False, compression="gzip"
    )
    rawtest_long.loc[:, [*sample_key_columns, *selected_features]].head(persist_limit).to_csv(
        rawtest_sample_path, index=False, compression="gzip"
    )
    contract = {
        "experiment": OUTPUT_PREFIX,
        "source_experiment": "exp248_candidate_perturbation_augmentation_for_likelihood_ranker",
        "candidate_names": [item.name for item in candidates],
        "train_base_feature_count": len(train_base_features),
        "rawtest_base_feature_count": len(raw_base_features),
        "train_long_feature_count": len(train_schema),
        "rawtest_long_feature_count": len(rawtest_schema),
        "selected_features": selected_features,
        "excluded_features": audit.loc[~audit["selected"], "feature"].tolist(),
        "decision": decision,
        "thresholds": get_nested(config, "feature_audit"),
    }
    _write_json(contract_path, contract)
    summary = {
        "status": "feature_audit_completed_ready_for_training"
        if decision["pass"]
        else "feature_audit_guard_failed",
        "execution_stage": "feature_audit_only",
        "runtime_seconds": time.time() - started,
        "train_rows": int(len(train_frame)),
        "train_wells": int(train_frame["well"].nunique()),
        "rawtest_rows": int(len(raw_frame)),
        "rawtest_wells": int(raw_frame["well"].nunique()),
        "candidate_count": len(candidates),
        "feature_decision": decision,
        "train_source_meta": train_source_meta,
        "rawtest_source_meta": raw_source_meta,
        "artifacts": {
            "feature_audit": audit_path.name,
            "selected_feature_schema": schema_path_out.name,
            "feature_contract": contract_path.name,
            "train_feature_sample": train_sample_path.name,
            "rawtest_feature_sample": rawtest_sample_path.name,
            "feature_audit_summary": summary_path.name,
        },
        "sha256": {
            "feature_audit": engine.sha256_path(audit_path),
            "selected_feature_schema": engine.sha256_path(schema_path_out),
            "feature_contract": engine.sha256_path(contract_path),
            "train_feature_sample_content": engine.sha256_path(
                train_sample_path, decompressed=True
            ),
            "rawtest_feature_sample_content": engine.sha256_path(
                rawtest_sample_path, decompressed=True
            ),
        },
    }
    _write_json(summary_path, summary)
    summary["sha256"]["feature_audit_summary"] = engine.sha256_path(summary_path)
    state = {
        "train_frame": train_frame,
        "candidates": candidates,
        "train_candidate_values": train_candidate_values,
        "train_base_features": train_base_features,
        "selected_features": selected_features,
        "feature_contract": contract,
    }
    return summary, state


def _training_decision(
    results: dict[str, pd.DataFrame], feature_decision: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    variants = [str(value) for value in get_nested(config, "model.active_variants") or []]
    if len(variants) != 1:
        raise ValueError(f"training decision requires exactly one active variant: {variants}")
    variant = variants[0]
    mode = "expected_error_fixed_viterbi"
    metric_mask = results["metrics"]["variant"].eq(variant) & results["metrics"]["mode"].eq(mode)
    selected = results["metrics"].loc[metric_mask].iloc[0]
    bucket_frame = results["bucket_metrics"]
    distance_mask = (
        bucket_frame["variant"].eq(variant)
        & bucket_frame["mode"].eq(mode)
        & bucket_frame["bucket_family"].eq("distance_bucket")
        & bucket_frame["bucket"].eq("1000_plus")
    )
    distance = bucket_frame.loc[distance_mask].iloc[0]
    subgroup_frame = results["subgroup_metrics"]
    subgroup_mask = subgroup_frame["variant"].eq(variant) & subgroup_frame["mode"].eq(mode)
    spatial = subgroup_frame.loc[
        subgroup_mask & subgroup_frame["subgroup"].eq("exp115_spatial_valid")
    ].iloc[0]
    typewell = subgroup_frame.loc[
        subgroup_mask & subgroup_frame["subgroup"].eq("exp115_typewell_purged_valid")
    ].iloc[0]
    by_well = results["by_well"]
    worst = (
        by_well.loc[by_well["variant"].eq(variant) & by_well["mode"].eq(mode)]
        .sort_values("rmse_tvt", ascending=False)
        .iloc[0]
    )
    criteria = get_nested(config, "audit.success_criteria") or {}
    checks = {
        "feature_audit_pass": bool(feature_decision["pass"]),
        "overall_vs_exp218": float(selected["rmse_tvt"]) <= float(criteria["max_selected_rmse"]),
        "distance_1000_plus": float(distance["rmse_tvt"])
        <= float(criteria["max_distance_1000_plus_rmse"]),
        "hidden_like_spatial": float(spatial["rmse_tvt"])
        <= float(criteria["max_hidden_like_spatial_rmse"]),
        "hidden_like_typewell_purged": float(typewell["rmse_tvt"])
        <= float(criteria["max_hidden_like_typewell_purged_rmse"]),
        "worst_well": float(worst["rmse_tvt"]) <= float(criteria["max_worst_well_rmse"]),
    }
    return {
        "adoption_supported": bool(all(checks.values())),
        "checks": checks,
        "selected_rmse": float(selected["rmse_tvt"]),
        "delta_vs_exp248_original_only": float(selected["rmse_tvt"])
        - float(criteria["exp248_original_only_rmse_reference"]),
        "distance_1000_plus_rmse": float(distance["rmse_tvt"]),
        "hidden_like_spatial_rmse": float(spatial["rmse_tvt"]),
        "hidden_like_typewell_purged_rmse": float(typewell["rmse_tvt"]),
        "worst_well": str(worst["well"]),
        "worst_well_rmse": float(worst["rmse_tvt"]),
    }


def run_training_after_feature_audit(
    *, output_dir: str | Path, audit_summary: dict[str, Any], state: dict[str, Any]
) -> dict[str, Any]:
    started = time.time()
    if not bool(state["feature_contract"]["decision"]["pass"]):
        raise RuntimeError(
            "10-booster training is blocked because the same-run feature audit failed"
        )
    config = load_config()
    output_dir = Path(output_dir)
    oof_scores, inventory, model_manifest, feature_schema, feature_importance = (
        engine.train_outer_oof_models(
            frame=state["train_frame"],
            candidates=state["candidates"],
            candidate_values=state["train_candidate_values"],
            base_feature_columns=state["train_base_features"],
            config=config,
            output_dir=output_dir,
            selected_feature_columns=state["selected_features"],
        )
    )
    if len(model_manifest) != 10:
        raise AssertionError(f"expected exactly 10 models, got {len(model_manifest)}")
    results = engine.evaluate_oof(
        frame=state["train_frame"],
        candidates=state["candidates"],
        candidate_values=state["train_candidate_values"],
        oof_scores=oof_scores,
        config=config,
    )
    decision = _training_decision(results, state["feature_contract"]["decision"], config)
    artifact_paths = {
        "metrics": output_dir / f"{OUTPUT_PREFIX}_metrics.csv",
        "candidate_metrics": output_dir / f"{OUTPUT_PREFIX}_candidate_metrics.csv",
        "calibration": output_dir / f"{OUTPUT_PREFIX}_calibration.csv",
        "topk_coverage": output_dir / f"{OUTPUT_PREFIX}_topk_coverage.csv",
        "margin_calibration": output_dir / f"{OUTPUT_PREFIX}_margin_calibration.csv",
        "oof_predictions": output_dir / f"{OUTPUT_PREFIX}_oof_predictions.csv.gz",
        "by_well": output_dir / f"{OUTPUT_PREFIX}_by_well.csv",
        "bucket_metrics": output_dir / f"{OUTPUT_PREFIX}_bucket_metrics.csv",
        "subgroup_metrics": output_dir / f"{OUTPUT_PREFIX}_subgroup_metrics.csv",
        "feature_importance_mean": output_dir / f"{OUTPUT_PREFIX}_feature_importance_mean.csv",
        "model_manifest": output_dir / f"{OUTPUT_PREFIX}_model_manifest.json",
        "summary": output_dir / f"{OUTPUT_PREFIX}_summary.json",
    }
    for key in [
        "metrics",
        "candidate_metrics",
        "calibration",
        "topk_coverage",
        "margin_calibration",
        "by_well",
        "bucket_metrics",
        "subgroup_metrics",
    ]:
        results[key].to_csv(artifact_paths[key], index=False)
    results["predictions"].to_csv(
        artifact_paths["oof_predictions"], index=False, compression="gzip"
    )
    _write_json(artifact_paths["model_manifest"], {"models": model_manifest.to_dict("records")})
    importance_mean = (
        feature_importance.groupby(["variant", "objective", "feature"], as_index=False)[
            "importance"
        ]
        .mean()
        .sort_values(["variant", "objective", "importance"], ascending=[True, True, False])
    )
    importance_mean.to_csv(artifact_paths["feature_importance_mean"], index=False)
    summary = {
        "status": "completed_train_side_adoption_supported"
        if decision["adoption_supported"]
        else "completed_train_side_guard_failed",
        "execution_stage": "train_after_feature_audit",
        "runtime_seconds": time.time() - started,
        "rows": int(len(state["train_frame"])),
        "wells": int(state["train_frame"]["well"].nunique()),
        "candidate_count": len(state["candidates"]),
        "selected_feature_count": len(feature_schema),
        "model_count": int(len(model_manifest)),
        "feature_audit": audit_summary,
        "decision": decision,
        "metrics": results["metrics"].to_dict("records"),
        "candidate_metrics": results["candidate_metrics"].to_dict("records"),
        "augmentation_inventory_rows": int(len(inventory)),
        "artifacts": {key: path.name for key, path in artifact_paths.items()},
    }
    summary["sha256"] = {
        key: engine.sha256_path(path, decompressed=path.suffix == ".gz")
        for key, path in artifact_paths.items()
        if key != "summary"
    }
    _write_json(artifact_paths["summary"], summary)
    summary["sha256"]["summary"] = engine.sha256_path(artifact_paths["summary"])
    return summary


def run_raw_test_safe_candidate_ranker(
    *,
    output_dir: str | Path,
    cache_path: str | Path | None = None,
    schema_path: str | Path | None = None,
) -> dict[str, Any]:
    config = load_config()
    stage = str(get_nested(config, "execution.stage"))
    allowed = {str(value) for value in get_nested(config, "execution.allowed_stages") or []}
    if stage not in allowed:
        raise ValueError(f"execution.stage must be one of {sorted(allowed)}, got {stage}")
    audit_summary, state = run_feature_audit(
        output_dir=output_dir,
        cache_path=cache_path,
        schema_path=schema_path,
    )
    if stage == "feature_audit_only":
        return audit_summary
    return run_training_after_feature_audit(
        output_dir=output_dir,
        audit_summary=audit_summary,
        state=state,
    )


__all__ = [
    "OUTPUT_PREFIX",
    "run_feature_audit",
    "run_raw_test_safe_candidate_ranker",
    "run_training_after_feature_audit",
    "synthetic_feature_audit_contract_test",
]
