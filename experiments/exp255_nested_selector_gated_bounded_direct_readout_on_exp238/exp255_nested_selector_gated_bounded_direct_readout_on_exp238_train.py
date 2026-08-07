# %% [markdown]
# # exp255 nested selector gated bounded direct readout — train OOF audit
#
# The frozen exp238 add-only OOF prediction is corrected explicitly toward the
# fold-held-out selector top1 candidate. Candidate selection and every gate are
# target-free. Truth is read only after all predictions have been frozen.

# %% [markdown]
# ## Contents
# 1. Imports and runtime helpers
# 2. Configuration and zero-training contract
# 3. Frozen candidate surface
# 4. exp238 final OOF and nested selector scores
# 5. Target-free gated bounded controller
# 6. Metrics and adoption guards
# 7. Generated artifacts and SHA summary

# %%
from __future__ import annotations

import gc
import gzip
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from IPython.display import display

EXPERIMENT = "exp255_nested_selector_gated_bounded_direct_readout_on_exp238"
PACKAGE_DIR = Path.cwd()
if not (PACKAGE_DIR / "config.yaml").exists():
    PACKAGE_DIR = Path(f"experiments/{EXPERIMENT}")
CONFIG = yaml.safe_load((PACKAGE_DIR / "config.yaml").read_text())
IS_KAGGLE = Path("/kaggle/working").exists()
OUTPUT_DIR = Path("/kaggle/working/artifacts") if IS_KAGGLE else PACKAGE_DIR / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STARTED_AT = time.time()


def cfg(key: str, default: Any = None) -> Any:
    value: Any = CONFIG
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def import_file(name: str, candidates: list[Path], *, reset_settings: bool = False):
    path = next((item for item in candidates if item.exists()), None)
    if path is None:
        raise FileNotFoundError(f"Cannot resolve {name}: {candidates}")
    if reset_settings:
        sys.modules.pop("settings", None)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path, *, decompressed: bool = False) -> str:
    digest = hashlib.sha256()
    opener = gzip.open if decompressed and path.suffix == ".gz" else Path.open
    with opener(path, "rb") as handle:  # type: ignore[arg-type]
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_named(filename: str, local: list[Path] | None = None) -> Path:
    candidates = list(local or [])
    if Path("/kaggle/input").exists():
        candidates.extend(Path("/kaggle/input").glob(f"**/{filename}"))
    candidates.extend([PACKAGE_DIR / filename, OUTPUT_DIR / filename])
    valid = [path for path in candidates if path.exists() and path.stat().st_size]
    if not valid:
        raise FileNotFoundError(filename)
    exact = {str(path.resolve()): path for path in valid}
    if len(exact) > 1:
        sizes = {path.stat().st_size for path in exact.values()}
        if len(sizes) > 1:
            raise ValueError(f"Ambiguous {filename}: {list(exact.values())}")
    return next(iter(exact.values()))


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value) if not isinstance(value, (str, bytes)) else False:
        return None
    return value


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        np.sqrt(
            np.mean(
                np.square(
                    prediction.astype(np.float64) - truth.astype(np.float64)
                )
            )
        )
    )


exp237 = import_file(
    "exp237_source",
    [
        PACKAGE_DIR / "exp237_source/hmm_exp226_candidate_selector_on_exp183.py",
        Path(
            "experiments/exp237_hmm_exp226_candidate_selector_on_exp183/"
            "hmm_exp226_candidate_selector_on_exp183.py"
        ),
    ],
    reset_settings=True,
)

# %% [markdown]
# ## 2. Configuration and zero-training contract

# %%
candidate_columns = [str(value) for value in cfg("model.candidates")]
profile_names = [str(value) for value in cfg("model.active_variants")]
profiles = cfg("model.profiles")
if int(cfg("model.model_configs")) != 0 or int(cfg("model.folds_trained")) != 0:
    raise ValueError("model/fold training must remain zero")
if int(cfg("model.boosters")) != 0 or bool(cfg("model.parent_control_retraining")):
    raise ValueError("booster or parent/control retraining is forbidden")
if set(profile_names) != set(profiles):
    raise ValueError("active profile/config mismatch")

print(
    json.dumps(
        {
            "active_audits": 1,
            "fixed_profiles": profile_names,
            "model_configs": 0,
            "folds_trained": 0,
            "boosters": 0,
            "parent_control_retraining": False,
            "runtime": "kaggle_cpu" if IS_KAGGLE else "static_only_local",
            "submission_requested": False,
        },
        indent=2,
    )
)

# %% [markdown]
# ## 3. Frozen candidate surface

# %%
parent_config = exp237.load_config()
parent_config.setdefault("inference", {})["use_test_base_as_dense_auxiliary"] = False
# Only columns needed to rebuild the candidate values are loaded. The frozen
# selector scores already contain the learned 184-feature context transform.
ranker = parent_config.setdefault("ranker", {})
ranker["context_columns"] = []
ranker["multiobs_feature_columns"] = []
ranker["optional_columns"] = []
ranker.setdefault("cluster_prior_features", {})["enabled"] = False

candidate_specs = exp237.candidate_specs_from_config(parent_config)
parent_candidate_columns = [item.column for item in candidate_specs]
if parent_candidate_columns != candidate_columns:
    raise ValueError(
        f"candidate schema mismatch: {parent_candidate_columns} != {candidate_columns}"
    )
required_columns = exp237.build_required_columns(parent_config, candidate_specs)
if "md_since" not in required_columns:
    required_columns.append("md_since")

frame, base_source_meta = exp237.load_train_feature_cache(
    cache_path=exp237.get_nested(parent_config, "data.exp099_train_feature_cache_local"),
    schema_path=exp237.get_nested(parent_config, "data.exp099_train_feature_schema_local"),
    required_columns=sorted(set(required_columns)),
    max_rows=None,
)
frame, _, enrichment_meta = exp237.add_feature_enrichment(
    frame, parent_config, max_rows=None
)
frame, _, external_meta = exp237.add_hmm_exp226_candidate_sources(
    frame, parent_config
)

if len(frame) != 3_783_989 or frame["well"].nunique() != 773:
    raise ValueError(
        f"candidate surface contract failed: rows={len(frame)} wells={frame['well'].nunique()}"
    )
if frame.duplicated(["id", "well"]).any():
    raise ValueError("candidate surface contains duplicate id/well")

candidate_values = frame[candidate_columns].to_numpy(np.float32, copy=True)
if not np.isfinite(candidate_values).all():
    raise ValueError("candidate surface contains non-finite values")
ids = frame["id"].astype(str).to_numpy()
wells = frame["well"].astype(str).to_numpy()
last_known_tvt = frame["last_known_tvt"].to_numpy(np.float32)
target_residual = frame["target"].to_numpy(np.float32)
md_since = frame["md_since"].to_numpy(np.float32)
keys = frame[["id", "well"]].astype(str).reset_index(drop=True)
del frame
gc.collect()

print(
    {
        "candidate_rows": len(ids),
        "candidate_wells": int(pd.Series(wells).nunique()),
        "candidate_count": len(candidate_columns),
        "candidate_all_finite": True,
    }
)

# %% [markdown]
# ## 4. exp238 final OOF and nested selector scores

# %%
final_oof_path = find_named(str(cfg("data.exp238_final_oof")))
final_prediction_column = str(cfg("data.exp238_final_prediction_column"))
final_header = pd.read_csv(final_oof_path, nrows=0).columns.tolist()
needed_final = ["id", "well", final_prediction_column]
missing_final = [column for column in needed_final if column not in final_header]
if missing_final:
    raise ValueError(f"exp238 final OOF missing columns: {missing_final}")
final_oof = pd.read_csv(
    final_oof_path,
    usecols=needed_final,
    dtype={"id": str, "well": str},
    low_memory=False,
)
if not final_oof[["id", "well"]].astype(str).reset_index(drop=True).equals(keys):
    raise ValueError("exp238 final OOF is not aligned with the candidate surface")
base_prediction = final_oof[final_prediction_column].to_numpy(np.float32)
if not np.isfinite(base_prediction).all():
    raise ValueError("exp238 final OOF contains non-finite predictions")
del final_oof
gc.collect()

selector_summary_path = find_named(str(cfg("data.exp238_selector_summary")))
selector_summary = json.loads(selector_summary_path.read_text())
score_columns = [f"pred_error__{name}" for name in candidate_columns]
n_rows = len(ids)
score_matrix = np.full((n_rows, len(candidate_columns)), np.nan, np.float32)
outer_fold = np.full(n_rows, -1, np.int8)
coverage = np.zeros(n_rows, np.uint8)
score_manifest: list[dict[str, Any]] = []
chunksize = int(cfg("audit.score_read_chunksize"))
score_prefix = str(cfg("data.exp238_selector_score_prefix"))

summary_score_items = selector_summary.get(
    "score_manifest", selector_summary.get("score_artifacts", [])
)
summary_score_sha = {
    int(item["outer_fold"]): str(item["sha256_decompressed"])
    for item in summary_score_items
    if "outer_fold" in item and "sha256_decompressed" in item
}
expected_score_sha = {
    int(fold): str(value)
    for fold, value in cfg(
        "data.exp238_selector_score_sha256_decompressed", {}
    ).items()
}
expected_folds = set(range(int(cfg("validation.outer_folds"))))
if set(expected_score_sha) != expected_folds:
    raise ValueError(
        "configured selector score SHA contract must cover every outer fold"
    )
if summary_score_sha and summary_score_sha != expected_score_sha:
    raise ValueError("selector summary SHA contract differs from exp255 config")

for fold in range(int(cfg("validation.outer_folds"))):
    filename = f"{score_prefix}{fold}.csv.gz"
    score_path = find_named(filename)
    header = pd.read_csv(score_path, nrows=0).columns.tolist()
    required_score = ["row_index", "role", "id", "well", *score_columns]
    missing_score = [column for column in required_score if column not in header]
    if missing_score:
        raise ValueError(f"{filename} missing columns: {missing_score}")
    valid_count = 0
    for chunk in pd.read_csv(
        score_path,
        usecols=required_score,
        dtype={"id": str, "well": str, "role": str},
        chunksize=chunksize,
        low_memory=False,
    ):
        valid = chunk.loc[chunk["role"].eq("valid")]
        if not len(valid):
            continue
        rows = valid["row_index"].to_numpy(np.int64)
        if (rows < 0).any() or (rows >= n_rows).any() or coverage[rows].any():
            raise ValueError(f"outer {fold}: duplicate or invalid valid row_index")
        expected_keys = keys.iloc[rows].reset_index(drop=True)
        actual_keys = valid[["id", "well"]].astype(str).reset_index(drop=True)
        if not actual_keys.equals(expected_keys):
            raise ValueError(f"outer {fold}: selector score key mismatch")
        values = valid[score_columns].to_numpy(np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"outer {fold}: selector score contains non-finite values")
        score_matrix[rows] = values
        outer_fold[rows] = np.int8(fold)
        coverage[rows] = 1
        valid_count += len(rows)
    actual_sha = sha256_file(score_path, decompressed=True)
    expected_sha = expected_score_sha[fold]
    if actual_sha != expected_sha:
        raise ValueError(f"outer {fold}: selector score SHA mismatch")
    score_manifest.append(
        {
            "outer_fold": fold,
            "path": str(score_path),
            "valid_rows": valid_count,
            "sha256_decompressed": actual_sha,
            "expected_sha256_decompressed": expected_sha,
            "sha_match": actual_sha == expected_sha,
        }
    )
    print(score_manifest[-1])

if not coverage.all() or not np.isfinite(score_matrix).all():
    raise ValueError("outer-valid selector scores do not cover every row")
folds_per_well = pd.DataFrame({"well": wells, "outer_fold": outer_fold}).groupby(
    "well"
)["outer_fold"].nunique()
if not folds_per_well.eq(1).all():
    raise ValueError("a well appears in multiple outer-valid folds")

# %% [markdown]
# ## 5. Target-free gated bounded controller

# %%
default_candidate = str(cfg("model.selector_default_candidate"))
default_code = candidate_columns.index(default_candidate)
top1_code = np.argmin(score_matrix, axis=1).astype(np.int16)
top1_score = score_matrix[np.arange(n_rows), top1_code]
top2_score = np.partition(score_matrix, 1, axis=1)[:, 1]
default_score = score_matrix[:, default_code]
predicted_gain = (default_score - top1_score).astype(np.float32)
predicted_margin = (top2_score - top1_score).astype(np.float32)
top1_value = candidate_values[np.arange(n_rows), top1_code]
top1_delta_from_base = (top1_value - base_prediction).astype(np.float32)


def build_profile_prediction(
    name: str, profile: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], pd.DataFrame]:
    row_eligible = (
        (top1_code != default_code)
        & (predicted_gain >= float(profile["min_predicted_gain"]))
        & (predicted_margin >= float(profile["min_predicted_margin"]))
        & (top1_score <= float(profile["max_top1_predicted_error"]))
    )
    eligible = pd.DataFrame(
        {
            "well": wells[row_eligible],
            "candidate_code": top1_code[row_eligible],
            "direction": np.sign(top1_delta_from_base[row_eligible]).astype(np.int8),
        }
    )
    all_wells = pd.Index(pd.unique(wells), name="well")
    if len(eligible):
        eligible_count = eligible.groupby("well").size().reindex(all_wells, fill_value=0)
        candidate_count = eligible.groupby(["well", "candidate_code"]).size()
        dominant = candidate_count.groupby(level=0).max().reindex(all_wells, fill_value=0)
        direction_count = eligible.groupby(["well", "direction"]).size()
        direction_dominant = (
            direction_count.groupby(level=0).max().reindex(all_wells, fill_value=0)
        )
        dominant_share = dominant / eligible_count.clip(lower=1)
        direction_consistency = direction_dominant / eligible_count.clip(lower=1)
    else:
        eligible_count = pd.Series(0, index=all_wells)
        dominant_share = pd.Series(0.0, index=all_wells)
        direction_consistency = pd.Series(0.0, index=all_wells)
    well_safe = (
        (eligible_count >= int(profile["min_eligible_rows_per_well"]))
        & (
            dominant_share
            >= float(profile["min_well_dominant_candidate_share"])
        )
        & (
            direction_consistency
            >= float(profile["min_well_direction_consistency"])
        )
    )
    safe_lookup = well_safe.to_dict()
    gate = row_eligible & np.fromiter(
        (bool(safe_lookup.get(str(well), False)) for well in wells),
        dtype=bool,
        count=n_rows,
    )
    alpha = float(profile["alpha"])
    delta_clip = float(profile["candidate_delta_clip"])
    bounded_delta = np.clip(top1_delta_from_base, -delta_clip, delta_clip)
    correction = np.where(gate, alpha * bounded_delta, 0.0).astype(np.float32)
    prediction = (base_prediction + correction).astype(np.float32)
    direction_violations = int(
        np.sum(correction.astype(np.float64) * top1_delta_from_base.astype(np.float64) < -1e-8)
    )
    max_abs_move = float(np.max(np.abs(correction)))
    if direction_violations:
        raise AssertionError(f"{name}: correction moved away from top1")
    if max_abs_move > float(profile["max_move"]) + 1e-5:
        raise AssertionError(f"{name}: max move exceeded configured cap")
    well_audit = pd.DataFrame(
        {
            "profile": name,
            "well": all_wells.astype(str),
            "eligible_rows": eligible_count.to_numpy(np.int64),
            "dominant_candidate_share": dominant_share.to_numpy(np.float64),
            "direction_consistency": direction_consistency.to_numpy(np.float64),
            "well_safe": well_safe.to_numpy(bool),
        }
    )
    summary = {
        "profile": name,
        "row_eligible": int(row_eligible.sum()),
        "row_gated": int(gate.sum()),
        "row_gated_rate": float(gate.mean()),
        "well_safe": int(well_safe.sum()),
        "max_abs_move": max_abs_move,
        "direction_violations": direction_violations,
        "alpha": alpha,
        "candidate_delta_clip": delta_clip,
        "max_move": float(profile["max_move"]),
    }
    return prediction, gate, summary, well_audit


predictions: dict[str, np.ndarray] = {
    "exp238_addonly_base": base_prediction,
    "hard_top1_diagnostic": top1_value.astype(np.float32),
}
gates: dict[str, np.ndarray] = {}
gate_records: list[dict[str, Any]] = []
well_gate_parts: list[pd.DataFrame] = []
for profile_name in profile_names:
    prediction, gate, gate_summary, well_audit = build_profile_prediction(
        profile_name, profiles[profile_name]
    )
    predictions[profile_name] = prediction
    gates[profile_name] = gate
    gate_records.append(gate_summary)
    well_gate_parts.append(well_audit)
    print(gate_summary)

well_gate_audit = pd.concat(well_gate_parts, ignore_index=True)

# %% [markdown]
# ## 6. Metrics and adoption guards

# %%
# Truth is materialized only after every profile prediction and gate have been
# frozen. No controller helper receives this array.
truth = (last_known_tvt + target_residual).astype(np.float32)
assignment_path = find_named(
    str(cfg("validation.hidden_like_assignment_file")),
    local=[
        PACKAGE_DIR / "inputs" / str(cfg("validation.hidden_like_assignment_file")),
        Path("experiments/exp237_hmm_exp226_candidate_selector_on_exp183/inputs")
        / str(cfg("validation.hidden_like_assignment_file")),
    ],
)
assignments = pd.read_csv(assignment_path, dtype={"well_id": str})
assignment = assignments.set_index("well_id")
spatial_role = pd.Series(wells).map(
    assignment["verification_like_spatial_role"]
).to_numpy()
typewell_role = pd.Series(wells).map(
    assignment["verification_like_typewell_purged_role"]
).to_numpy()
if pd.isna(spatial_role).any() or pd.isna(typewell_role).any():
    raise ValueError("hidden-like assignments do not cover every well")

scope_masks: dict[str, np.ndarray] = {"global": np.ones(n_rows, bool)}
for bucket in cfg("validation.distance_buckets"):
    name = str(bucket["name"])
    lower = float(bucket["min_md_since"])
    upper = bucket.get("max_md_since")
    mask = md_since >= lower
    if upper is not None:
        mask &= md_since < float(upper)
    scope_masks[name] = mask
scope_masks["exp115_spatial_valid"] = spatial_role == "valid"
scope_masks["exp115_typewell_purged_valid"] = typewell_role == "valid"
for fold in range(int(cfg("validation.outer_folds"))):
    scope_masks[f"outer_fold_{fold}"] = outer_fold == fold

metric_records: list[dict[str, Any]] = []
base_scope_rmse = {
    scope: rmse(truth[mask], base_prediction[mask])
    for scope, mask in scope_masks.items()
    if mask.any()
}
for variant, prediction in predictions.items():
    if not np.isfinite(prediction).all():
        raise ValueError(f"{variant} contains non-finite predictions")
    for scope, mask in scope_masks.items():
        if not mask.any():
            continue
        value = rmse(truth[mask], prediction[mask])
        metric_records.append(
            {
                "variant": variant,
                "scope": scope,
                "rows": int(mask.sum()),
                "rmse": value,
                "base_rmse": base_scope_rmse[scope],
                "delta_rmse": value - base_scope_rmse[scope],
            }
        )
metrics = pd.DataFrame(metric_records)

well_codes, unique_wells = pd.factorize(wells, sort=True)
well_counts = np.bincount(well_codes)
base_well_rmse = np.sqrt(
    np.bincount(
        well_codes,
        weights=np.square(
            base_prediction.astype(np.float64) - truth.astype(np.float64)
        ),
    )
    / well_counts
)
by_well_parts: list[pd.DataFrame] = []
for variant, prediction in predictions.items():
    variant_rmse = np.sqrt(
        np.bincount(
            well_codes,
            weights=np.square(
                prediction.astype(np.float64) - truth.astype(np.float64)
            ),
        )
        / well_counts
    )
    by_well_parts.append(
        pd.DataFrame(
            {
                "variant": variant,
                "well": unique_wells.astype(str),
                "rows": well_counts,
                "base_rmse": base_well_rmse,
                "rmse": variant_rmse,
                "delta_rmse": variant_rmse - base_well_rmse,
            }
        )
    )
by_well = pd.concat(by_well_parts, ignore_index=True)


def metric_delta(variant: str, scope: str) -> float:
    values = metrics.loc[
        metrics["variant"].eq(variant) & metrics["scope"].eq(scope),
        "delta_rmse",
    ]
    if len(values) != 1:
        raise ValueError(f"missing metric {variant}/{scope}")
    return float(values.iloc[0])


guard_cfg = cfg("validation.adoption_guard")
guard_results: dict[str, dict[str, Any]] = {}
for profile_name in profile_names:
    fold_deltas = [
        metric_delta(profile_name, f"outer_fold_{fold}")
        for fold in range(int(cfg("validation.outer_folds")))
    ]
    worst_well = float(
        by_well.loc[by_well["variant"].eq(profile_name), "delta_rmse"].max()
    )
    gate_summary = next(
        item for item in gate_records if item["profile"] == profile_name
    )
    checks = {
        "global_improvement": metric_delta(profile_name, "global") < 0.0,
        "near_nonworse": metric_delta(profile_name, "000_050") <= 0.0,
        "longtail_nonworse": metric_delta(profile_name, "1000_plus") <= 0.0,
        "hidden_spatial_nonworse": metric_delta(
            profile_name, "exp115_spatial_valid"
        )
        <= 0.0,
        "hidden_typewell_nonworse": metric_delta(
            profile_name, "exp115_typewell_purged_valid"
        )
        <= 0.0,
        "fold_improvement": int(np.sum(np.asarray(fold_deltas) < 0.0))
        >= int(guard_cfg["min_improved_outer_folds"]),
        "worst_well": worst_well
        <= float(guard_cfg["max_worst_well_regression"]),
        "zero_direction_violations": int(gate_summary["direction_violations"]) == 0,
        "move_cap": float(gate_summary["max_abs_move"])
        <= float(gate_summary["max_move"]) + 1e-5,
    }
    guard_results[profile_name] = {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "global_delta_rmse": metric_delta(profile_name, "global"),
        "near_000_050_delta_rmse": metric_delta(profile_name, "000_050"),
        "longtail_1000_plus_delta_rmse": metric_delta(profile_name, "1000_plus"),
        "hidden_spatial_delta_rmse": metric_delta(
            profile_name, "exp115_spatial_valid"
        ),
        "hidden_typewell_delta_rmse": metric_delta(
            profile_name, "exp115_typewell_purged_valid"
        ),
        "outer_fold_deltas": fold_deltas,
        "improved_outer_folds": int(np.sum(np.asarray(fold_deltas) < 0.0)),
        "worst_well_regression": worst_well,
    }

passing = [name for name in profile_names if guard_results[name]["pass"]]
global_profile_rmse = {
    name: float(
        metrics.loc[
            metrics["variant"].eq(name) & metrics["scope"].eq("global"),
            "rmse",
        ].iloc[0]
    )
    for name in profile_names
}
selected_profile = min(
    passing if passing else profile_names,
    key=lambda name: global_profile_rmse[name],
)
adoption_allowed = bool(passing)
status = (
    "oof_adoption_guard_passed_inference_not_implemented"
    if adoption_allowed
    else "oof_complete_no_profile_passed_inference_forbidden"
)

for record in gate_records:
    profile_name = str(record["profile"])
    record["adoption_guard_pass"] = bool(guard_results[profile_name]["pass"])
    record["worst_well_regression"] = float(
        guard_results[profile_name]["worst_well_regression"]
    )

candidate_distribution = (
    pd.DataFrame(
        {
            "candidate_code": top1_code,
            "outer_fold": outer_fold,
        }
    )
    .groupby(["outer_fold", "candidate_code"])
    .size()
    .rename("rows")
    .reset_index()
)
candidate_distribution["candidate"] = candidate_distribution["candidate_code"].map(
    dict(enumerate(candidate_columns))
)

# %% [markdown]
# ## 7. Generated artifacts and SHA summary

# %%
prefix = str(cfg("audit.output_prefix"))
metrics_path = OUTPUT_DIR / f"{prefix}_metrics.csv"
by_well_path = OUTPUT_DIR / f"{prefix}_by_well.csv"
gate_path = OUTPUT_DIR / f"{prefix}_gate_summary.csv"
candidate_path = OUTPUT_DIR / f"{prefix}_candidate_distribution.csv"
well_gate_path = OUTPUT_DIR / f"{prefix}_well_gate_audit.csv"
oof_path = OUTPUT_DIR / f"{prefix}_selected_oof.csv.gz"
manifest_path = OUTPUT_DIR / f"{prefix}_input_manifest.json"
summary_path = OUTPUT_DIR / f"{prefix}_summary.json"
plot_path = OUTPUT_DIR / f"{prefix}_global_delta.png"

metrics.to_csv(metrics_path, index=False)
by_well.to_csv(by_well_path, index=False)
pd.DataFrame(gate_records).to_csv(gate_path, index=False)
candidate_distribution.to_csv(candidate_path, index=False)
well_gate_audit.to_csv(well_gate_path, index=False)

selected_prediction = predictions[selected_profile]
selected_gate = gates[selected_profile]
selected_oof = pd.DataFrame(
    {
        "id": ids,
        "well": wells,
        "outer_fold": outer_fold,
        "md_since": md_since,
        "truth_tvt": truth,
        "base_pred_tvt": base_prediction,
        "top1_candidate_code": top1_code,
        "top1_candidate_tvt": top1_value,
        "top1_predicted_error": top1_score,
        "default_predicted_error": default_score,
        "predicted_gain": predicted_gain,
        "predicted_margin": predicted_margin,
        "gate": selected_gate.astype(np.int8),
        "prediction_tvt": selected_prediction,
    }
)
selected_oof.to_csv(oof_path, index=False, compression="gzip")

input_manifest = {
    "config": {
        "path": str(PACKAGE_DIR / "config.yaml"),
        "sha256": sha256_file(PACKAGE_DIR / "config.yaml"),
    },
    "candidate_base": base_source_meta,
    "candidate_enrichment": enrichment_meta,
    "candidate_external": external_meta,
    "exp238_final_oof": {
        "path": str(final_oof_path),
        "sha256": sha256_file(final_oof_path),
        "sha256_decompressed": sha256_file(final_oof_path, decompressed=True),
    },
    "exp238_selector_summary": {
        "path": str(selector_summary_path),
        "sha256": sha256_file(selector_summary_path),
    },
    "exp238_selector_scores": score_manifest,
    "hidden_like_assignment": {
        "path": str(assignment_path),
        "sha256": sha256_file(assignment_path),
    },
}
manifest_path.write_text(
    json.dumps(to_jsonable(input_manifest), indent=2, sort_keys=True) + "\n"
)

global_metrics = metrics.loc[metrics["scope"].eq("global")].copy()
plt.figure(figsize=(9, 4.5))
plt.bar(global_metrics["variant"], global_metrics["delta_rmse"])
plt.axhline(0.0, color="black", linewidth=1)
plt.xticks(rotation=25, ha="right")
plt.ylabel("RMSE delta vs exp238 add-only")
plt.title("exp255 fixed gated/bounded profiles")
plt.tight_layout()
plt.savefig(plot_path, dpi=150)
plt.close()

summary = {
    "status": status,
    "experiment": EXPERIMENT,
    "runtime_seconds": float(time.time() - STARTED_AT),
    "rows": n_rows,
    "wells": int(len(unique_wells)),
    "outer_folds": int(cfg("validation.outer_folds")),
    "candidate_count": len(candidate_columns),
    "candidate_columns": candidate_columns,
    "fixed_profiles": profile_names,
    "model_configs": 0,
    "folds_trained": 0,
    "boosters": 0,
    "parent_control_retraining": False,
    "truth_used_in_gate": False,
    "role_valid_score_coverage": int(coverage.sum()),
    "hard_top1_diagnostic_rmse": rmse(truth, predictions["hard_top1_diagnostic"]),
    "base_rmse": rmse(truth, base_prediction),
    "selected_profile": selected_profile,
    "selected_profile_rmse": rmse(truth, selected_prediction),
    "adoption_allowed": adoption_allowed,
    "passing_profiles": passing,
    "guard_results": guard_results,
    "gate_summary": gate_records,
    "inference_executed": False,
    "submission_generated": False,
    "competition_submit_executed": False,
    "artifacts": {
        "metrics": metrics_path.name,
        "by_well": by_well_path.name,
        "gate_summary": gate_path.name,
        "candidate_distribution": candidate_path.name,
        "well_gate_audit": well_gate_path.name,
        "selected_oof": oof_path.name,
        "input_manifest": manifest_path.name,
        "plot": plot_path.name,
    },
    "sha256": {
        "metrics": sha256_file(metrics_path),
        "by_well": sha256_file(by_well_path),
        "gate_summary": sha256_file(gate_path),
        "candidate_distribution": sha256_file(candidate_path),
        "well_gate_audit": sha256_file(well_gate_path),
        "selected_oof_decompressed": sha256_file(oof_path, decompressed=True),
        "input_manifest": sha256_file(manifest_path),
    },
}
summary_path.write_text(
    json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n"
)

print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
display(global_metrics)
display(pd.DataFrame(gate_records))
