from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


OUTPUT_PREFIX = "exp148_feature_correlation_audit"
DEFAULT_SAMPLE_ROWS = 600_000
RANDOM_SEED = 148
CHUNKSIZE = 500_000
CORR_THRESHOLDS = (0.90, 0.95, 0.98, 0.995)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    return value


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError("None of these paths exists:\n" + "\n".join(str(p) for p in paths))


def _kernel_source_dir(slug: str) -> Path:
    candidates = [
        Path("/kaggle/input/notebooks/kentookumura") / slug,
        Path("/kaggle/input") / slug,
    ]
    for path in candidates:
        if path.exists():
            return path

    notebooks_root = Path("/kaggle/input/notebooks/kentookumura")
    if notebooks_root.exists():
        matches = sorted(path for path in notebooks_root.iterdir() if slug[:8] in path.name)
        if matches:
            return matches[0]
    input_matches = sorted(path for path in Path("/kaggle/input").iterdir() if slug[:8] in path.name)
    if input_matches:
        return input_matches[0]
    raise FileNotFoundError(f"Kaggle input source not found for slug={slug}")


def _upper_triangle_pairs(corr: pd.DataFrame, schema: pd.DataFrame) -> pd.DataFrame:
    features = corr.columns.to_numpy()
    arr = corr.to_numpy(np.float64)
    rows: list[dict[str, Any]] = []
    feature_to_family = dict(zip(schema["feature"], schema["family"], strict=False))
    for i in range(len(features)):
        values = arr[i, i + 1 :]
        if values.size == 0:
            continue
        right_features = features[i + 1 :]
        finite = np.isfinite(values)
        if not finite.any():
            continue
        for feature_b, value in zip(right_features[finite], values[finite], strict=False):
            rows.append(
                {
                    "feature_a": str(features[i]),
                    "feature_b": str(feature_b),
                    "corr": float(value),
                    "abs_corr": float(abs(value)),
                    "family_a": feature_to_family.get(str(features[i]), "unknown"),
                    "family_b": feature_to_family.get(str(feature_b), "unknown"),
                }
            )
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False).reset_index(drop=True)


def _connected_components(pairs: pd.DataFrame, threshold: float) -> list[list[str]]:
    adjacency: dict[str, set[str]] = {}
    for row in pairs[pairs["abs_corr"].ge(threshold)].itertuples(index=False):
        adjacency.setdefault(str(row.feature_a), set()).add(str(row.feature_b))
        adjacency.setdefault(str(row.feature_b), set()).add(str(row.feature_a))

    seen: set[str] = set()
    components: list[list[str]] = []
    for start in sorted(adjacency):
        if start in seen:
            continue
        stack = [start]
        component: list[str] = []
        seen.add(start)
        while stack:
            node = stack.pop()
            component.append(node)
            for nxt in sorted(adjacency.get(node, set())):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        if len(component) > 1:
            components.append(sorted(component))
    return components


def _component_table(
    pairs: pd.DataFrame,
    feature_readout: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    importance = dict(zip(feature_readout["feature"], feature_readout["mean_importance"], strict=False))
    family = dict(zip(feature_readout["feature"], feature_readout["family"], strict=False))
    rows: list[dict[str, Any]] = []
    for component_id, component in enumerate(_connected_components(pairs, threshold), start=1):
        ranked = sorted(
            component,
            key=lambda feature: (-float(importance.get(feature, 0.0)), feature),
        )
        component_pairs = pairs[
            pairs["feature_a"].isin(component) & pairs["feature_b"].isin(component)
        ]
        rows.append(
            {
                "component_id": component_id,
                "threshold": threshold,
                "n_features": len(component),
                "keeper_by_importance": ranked[0],
                "drop_candidates": "|".join(ranked[1:]),
                "features": "|".join(ranked),
                "families": "|".join(sorted({str(family.get(feature, "unknown")) for feature in component})),
                "keeper_importance": float(importance.get(ranked[0], 0.0)),
                "drop_importance_sum": float(sum(float(importance.get(feature, 0.0)) for feature in ranked[1:])),
                "max_abs_corr": float(component_pairs["abs_corr"].max()),
                "min_abs_corr_in_component_pairs": float(component_pairs["abs_corr"].min()),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "component_id",
                "threshold",
                "n_features",
                "keeper_by_importance",
                "drop_candidates",
                "features",
                "families",
                "keeper_importance",
                "drop_importance_sum",
                "max_abs_corr",
                "min_abs_corr_in_component_pairs",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["n_features", "keeper_importance"], ascending=[False, False]
    )


def _feature_family(schema: pd.DataFrame) -> pd.Series:
    is_projection = schema["is_projection_feature"].astype(str).str.lower().eq("true")
    is_learned = schema["is_learned_likelihood_feature"].astype(str).str.lower().eq("true")
    family = pd.Series("base_replay", index=schema.index, dtype=object)
    family.loc[is_projection] = "u_projection"
    family.loc[is_learned] = "learned_likelihood"
    return family


def main() -> None:
    t0 = time.time()
    output_dir = Path("/kaggle/working")
    sample_rows = int(os.environ.get("EXP148_CORR_SAMPLE_ROWS", DEFAULT_SAMPLE_ROWS))
    rng = np.random.default_rng(RANDOM_SEED)

    exp148_dir = _kernel_source_dir("exp148-train")
    exp072_dir = _kernel_source_dir("exp072-exp063-full-replay-feature-cache-train")
    exp145_dir = _kernel_source_dir("exp145-train")

    sys.path.insert(0, str(exp148_dir))
    from learned_likelihood_fulltrain_addonly_on_exp092 import (  # noqa: PLC0415
        add_anchor_columns,
        build_learned_likelihood_features,
        build_u_projection_features,
    )
    from settings import ExperimentPaths  # noqa: PLC0415

    config_path = exp148_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    projection_config = dict(config["model"]["u_projection"])
    learned_config = dict(config["model"]["learned_likelihood_features"])

    base_path = _first_existing(
        [
            exp072_dir
            / "artifacts"
            / "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz",
            exp072_dir / "exp063_full_replay_feature_cache_pixiux_likpf_public_replay_train_features.csv.gz",
        ]
    )
    learned_path = _first_existing(
        [
            exp145_dir
            / "artifacts"
            / "exp145_learned_likelihood_rawtest_feature_generator_parity_full_train_ml_features.csv.gz",
            exp145_dir
            / "kaggle"
            / "output"
            / "train_v2"
            / "artifacts"
            / "exp145_learned_likelihood_rawtest_feature_generator_parity_full_train_ml_features.csv.gz",
        ]
    )
    feature_schema_path = _first_existing(
        [
            exp148_dir
            / "artifacts"
            / "exp148_learned_likelihood_fulltrain_addonly_on_exp092_feature_schema.csv",
            exp148_dir / "exp148_learned_likelihood_fulltrain_addonly_on_exp092_feature_schema.csv",
        ]
    )
    importance_path = _first_existing(
        [
            exp148_dir
            / "artifacts"
            / "exp148_learned_likelihood_fulltrain_addonly_on_exp092_feature_importance_mean.csv",
            exp148_dir
            / "exp148_learned_likelihood_fulltrain_addonly_on_exp092_feature_importance_mean.csv",
        ]
    )
    train_dir = ExperimentPaths().train_data_dir
    if not train_dir.exists():
        raise FileNotFoundError(f"Resolved train_data_dir does not exist: {train_dir}")

    schema = pd.read_csv(feature_schema_path)
    schema["feature"] = schema["feature"].astype(str)
    schema["family"] = _feature_family(schema)
    feature_columns = schema["feature"].tolist()
    base_feature_columns = schema[
        schema["family"].eq("base_replay")
    ]["feature"].tolist()
    learned_feature_columns = schema[
        schema["family"].eq("learned_likelihood")
    ]["feature"].tolist()
    projection_feature_columns = schema[
        schema["family"].eq("u_projection")
    ]["feature"].tolist()

    base_usecols = ["id", "well", "target", *base_feature_columns]
    print(
        json.dumps(
            {
                "event": "read_base_cache",
                "path": str(base_path),
                "base_usecols": len(base_usecols),
            }
        ),
        flush=True,
    )
    base = pd.read_csv(base_path, usecols=base_usecols, dtype={"id": str, "well": str})
    for col in base.columns:
        if col not in {"id", "well"}:
            base[col] = pd.to_numeric(base[col], errors="coerce").astype(np.float32)
    if not np.isfinite(base[["target", *base_feature_columns]].to_numpy(np.float32)).all():
        raise ValueError("base cache contains non-finite values")

    print(
        json.dumps({"event": "add_anchor_columns", "rows": len(base), "wells": base["well"].nunique()}),
        flush=True,
    )
    base, anchor_meta = add_anchor_columns(base, train_dir)

    print(json.dumps({"event": "build_projection_features"}), flush=True)
    projection_features, _, projection_summary = build_u_projection_features(
        base,
        source_specs=dict(projection_config.get("sources") or {}),
        degree=int(projection_config.get("degree", 3)),
        robust_iters=int(projection_config.get("robust_iters", 3)),
        clip_sigma=float(projection_config.get("clip_sigma", 4.0)),
    )

    n_rows = len(base)
    sample_size = min(sample_rows, n_rows)
    sample_idx = np.sort(rng.choice(n_rows, size=sample_size, replace=False))
    base_sample = base.iloc[sample_idx].copy().reset_index(drop=True)
    projection_sample = (
        projection_features.iloc[sample_idx][projection_feature_columns]
        .copy()
        .reset_index(drop=True)
    )
    sample_frame = pd.concat([base_sample, projection_sample], axis=1)
    del base, projection_features, projection_sample
    gc.collect()

    requested_learned_source_cols = [
        *learned_config.get("direct_columns", []),
        *learned_config.get("weighted_tvt_columns", []),
        *learned_config.get("candidate_tvt_columns", []),
    ]
    learned_usecols = ["id", "well", *requested_learned_source_cols]
    sample_ids = set(sample_frame["id"].astype(str).tolist())
    learned_chunks: list[pd.DataFrame] = []
    print(
        json.dumps(
            {
                "event": "read_learned_cache_for_sample",
                "path": str(learned_path),
                "usecols": len(learned_usecols),
                "sample_ids": len(sample_ids),
            }
        ),
        flush=True,
    )
    for chunk in pd.read_csv(
        learned_path,
        usecols=learned_usecols,
        dtype={"id": str, "well": str},
        chunksize=CHUNKSIZE,
    ):
        selected = chunk[chunk["id"].isin(sample_ids)].copy()
        if not selected.empty:
            for col in requested_learned_source_cols:
                selected[col] = pd.to_numeric(selected[col], errors="coerce").astype(np.float32)
            learned_chunks.append(selected)
    if not learned_chunks:
        raise ValueError("No learned likelihood rows matched the sampled base rows")
    learned_source = pd.concat(learned_chunks, ignore_index=True)
    del learned_chunks
    gc.collect()

    learned_features, _, learned_summary = build_learned_likelihood_features(
        learned_source,
        sample_frame,
        learned_config,
    )
    del learned_source
    gc.collect()

    sample_frame = sample_frame.merge(
        learned_features,
        on=["id", "well"],
        how="inner",
        validate="one_to_one",
    )
    if len(sample_frame) != sample_size:
        raise ValueError(
            f"sample join dropped rows: sample_size={sample_size}, joined={len(sample_frame)}"
        )

    missing_features = sorted(set(feature_columns) - set(sample_frame.columns))
    if missing_features:
        raise ValueError(f"sample frame is missing model features: {missing_features[:20]}")

    print(json.dumps({"event": "compute_correlations", "rows": len(sample_frame)}), flush=True)
    feature_matrix = sample_frame[feature_columns].astype(np.float32)
    target = sample_frame["target"].astype(np.float32)
    corr = feature_matrix.corr(method="pearson")
    corr.to_csv(output_dir / f"{OUTPUT_PREFIX}_corr_matrix.csv.gz", compression="gzip")

    target_corr = feature_matrix.corrwith(target, method="pearson")
    std = feature_matrix.std(axis=0)
    nunique = feature_matrix.nunique(dropna=False)

    pairs = _upper_triangle_pairs(corr, schema)
    pairs.to_csv(output_dir / f"{OUTPUT_PREFIX}_all_corr_pairs.csv.gz", index=False, compression="gzip")
    pairs[pairs["abs_corr"].ge(0.90)].to_csv(
        output_dir / f"{OUTPUT_PREFIX}_pairs_abs_ge_090.csv",
        index=False,
    )
    pairs.head(500).to_csv(output_dir / f"{OUTPUT_PREFIX}_top500_pairs.csv", index=False)

    importance = pd.read_csv(importance_path)
    importance = importance[["feature", "mean_importance", "std_importance", "fold_model_records"]]
    importance["feature"] = importance["feature"].astype(str)

    max_partner_rows: list[dict[str, Any]] = []
    for feature in feature_columns:
        subset = pairs[pairs["feature_a"].eq(feature) | pairs["feature_b"].eq(feature)]
        if subset.empty:
            max_partner_rows.append(
                {
                    "feature": feature,
                    "max_abs_corr": np.nan,
                    "max_corr": np.nan,
                    "max_corr_partner": None,
                }
            )
            continue
        best = subset.iloc[0]
        partner = best["feature_b"] if best["feature_a"] == feature else best["feature_a"]
        max_partner_rows.append(
            {
                "feature": feature,
                "max_abs_corr": float(best["abs_corr"]),
                "max_corr": float(best["corr"]),
                "max_corr_partner": str(partner),
            }
        )
    max_partner = pd.DataFrame(max_partner_rows)

    readout = (
        schema[["feature", "feature_index", "family"]]
        .merge(importance, on="feature", how="left")
        .merge(max_partner, on="feature", how="left")
    )
    readout["target_corr"] = readout["feature"].map(target_corr.to_dict())
    readout["abs_target_corr"] = readout["target_corr"].abs()
    readout["std"] = readout["feature"].map(std.to_dict())
    readout["nunique"] = readout["feature"].map(nunique.to_dict())
    readout["mean_importance"] = readout["mean_importance"].fillna(0.0)
    readout["importance_rank"] = readout["mean_importance"].rank(
        method="dense", ascending=False
    ).astype(int)

    components_098 = _component_table(pairs, readout, 0.98)
    components_0995 = _component_table(pairs, readout, 0.995)
    components_098.to_csv(output_dir / f"{OUTPUT_PREFIX}_components_abs_ge_098.csv", index=False)
    components_0995.to_csv(output_dir / f"{OUTPUT_PREFIX}_components_abs_ge_0995.csv", index=False)

    prune_map: dict[str, int] = {}
    keeper_map: dict[str, str] = {}
    for row in components_098.itertuples(index=False):
        drops = [item for item in str(row.drop_candidates).split("|") if item]
        for feature in drops:
            prune_map[feature] = int(row.component_id)
            keeper_map[feature] = str(row.keeper_by_importance)
    readout["component_abs_ge_098"] = readout["feature"].map(prune_map)
    readout["keeper_if_pruned_abs_ge_098"] = readout["feature"].map(keeper_map)
    readout["prune_candidate_abs_ge_098"] = readout["component_abs_ge_098"].notna()
    readout = readout.sort_values(
        ["prune_candidate_abs_ge_098", "max_abs_corr", "mean_importance"],
        ascending=[False, False, True],
    )
    readout.to_csv(output_dir / f"{OUTPUT_PREFIX}_feature_readout.csv", index=False)

    pair_counts = {str(th): int(pairs["abs_corr"].ge(th).sum()) for th in CORR_THRESHOLDS}
    family_pair_counts = (
        pairs[pairs["abs_corr"].ge(0.98)]
        .assign(family_pair=lambda frame: frame["family_a"] + "__" + frame["family_b"])
        .groupby("family_pair")
        .size()
        .sort_values(ascending=False)
        .to_dict()
    )
    summary = {
        "experiment": "exp148_learned_likelihood_fulltrain_addonly_on_exp092",
        "status": "completed",
        "sample_policy": {
            "rows_total": int(n_rows),
            "sample_rows_requested": int(sample_rows),
            "sample_rows_used": int(len(sample_frame)),
            "seed": RANDOM_SEED,
            "sampling": "global_uniform_without_replacement_after_full-row projection feature construction",
        },
        "feature_counts": {
            "total": int(len(feature_columns)),
            "base_replay": int(len(base_feature_columns)),
            "u_projection": int(len(projection_feature_columns)),
            "learned_likelihood": int(len(learned_feature_columns)),
        },
        "pair_counts_by_abs_corr_threshold": pair_counts,
        "pair_counts_abs_ge_098_by_family_pair": _jsonable(family_pair_counts),
        "components_abs_ge_098": int(len(components_098)),
        "features_in_components_abs_ge_098": int(
            sum(len(str(row.features).split("|")) for row in components_098.itertuples(index=False))
        ),
        "prune_candidates_abs_ge_098": int(readout["prune_candidate_abs_ge_098"].sum()),
        "constant_or_single_value_features": readout.loc[
            readout["nunique"].le(1), "feature"
        ].tolist(),
        "top_abs_corr_pairs": _jsonable(pairs.head(30).to_dict("records")),
        "top_importance_features": _jsonable(
            readout.sort_values("mean_importance", ascending=False)
            .head(30)[
                [
                    "feature",
                    "family",
                    "mean_importance",
                    "max_abs_corr",
                    "max_corr_partner",
                    "target_corr",
                ]
            ]
            .to_dict("records")
        ),
        "largest_components_abs_ge_098": _jsonable(
            components_098.head(20).to_dict("records")
        ),
        "anchor_meta": _jsonable(anchor_meta),
        "projection_summary": _jsonable(projection_summary.to_dict("records")),
        "learned_summary": _jsonable(learned_summary.to_dict("records")),
        "outputs": {
            "feature_readout": f"{OUTPUT_PREFIX}_feature_readout.csv",
            "pairs_abs_ge_090": f"{OUTPUT_PREFIX}_pairs_abs_ge_090.csv",
            "top500_pairs": f"{OUTPUT_PREFIX}_top500_pairs.csv",
            "components_abs_ge_098": f"{OUTPUT_PREFIX}_components_abs_ge_098.csv",
            "components_abs_ge_0995": f"{OUTPUT_PREFIX}_components_abs_ge_0995.csv",
            "all_corr_pairs": f"{OUTPUT_PREFIX}_all_corr_pairs.csv.gz",
            "corr_matrix": f"{OUTPUT_PREFIX}_corr_matrix.csv.gz",
        },
        "elapsed_seconds": round(time.time() - t0, 3),
    }
    summary_path = output_dir / f"{OUTPUT_PREFIX}_summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2))
    print(json.dumps(_jsonable(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
