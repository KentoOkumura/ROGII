# %% [markdown]
# # exp258 GR residual augmentation selector train (CPU)

# %% [markdown]
# ## Contents
# 1. Imports and runtime contract
# 2. Fixed exp237 candidate surface
# 3. Fold-safe residual profile audit
# 4. Synthetic GR view and multi-observation feature rebuild
# 5. Strict nested selector training
# 6. Historical calibration and safety guard
# 7. Artifacts and decision

# %%
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from IPython.display import display
from sklearn.metrics import roc_auc_score

PACKAGE_DIR = Path.cwd()
if not (PACKAGE_DIR / "config.yaml").exists():
    PACKAGE_DIR = Path("experiments/exp258_gr_residual_noise_transplant_augmentation")
CONFIG = yaml.safe_load((PACKAGE_DIR / "config.yaml").read_text())
OUTPUT_PREFIX = str(CONFIG["experiment"]["name"])
OUTPUT_DIR = (
    Path("/kaggle/working/artifacts")
    if Path("/kaggle/working").exists()
    else PACKAGE_DIR / "artifacts"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def import_file(name: str, candidates: list[Path], *, reset_settings: bool = False):
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(f"Cannot resolve {name}: {candidates}")
    if reset_settings:
        sys.modules.pop("settings", None)
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


engine = import_file(
    "exp238_engine",
    [
        Path("experiments/exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218")
        / "nested_hmm_exp226_selector_rank_slot_addonly_on_exp218.py",
        PACKAGE_DIR / "exp238_source/nested_hmm_exp226_selector_rank_slot_addonly_on_exp218.py",
        PACKAGE_DIR / "nested_hmm_exp226_selector_rank_slot_addonly_on_exp218.py",
    ],
)
engine.OUTPUT_PREFIX = OUTPUT_PREFIX
exp237 = import_file(
    "exp237_source",
    [
        Path("experiments/exp237_hmm_exp226_candidate_selector_on_exp183")
        / "hmm_exp226_candidate_selector_on_exp183.py",
        Path("/kaggle/input/exp237-hmm-exp226-candidate-selector-exp183-train")
        / "hmm_exp226_candidate_selector_on_exp183.py",
        PACKAGE_DIR / "exp237_source/hmm_exp226_candidate_selector_on_exp183.py",
        PACKAGE_DIR / "hmm_exp226_candidate_selector_on_exp183.py",
    ],
    reset_settings=True,
)

for root in (Path.cwd(), PACKAGE_DIR, *PACKAGE_DIR.parents):
    if (root / "src/gr_residual_noise_augmentation.py").exists():
        sys.path.insert(0, str(root))
        break
from src.gr_residual_noise_augmentation import (  # noqa: E402
    content_sha256,
    profile_inventory,
    read_residual_profile,
    stable_uint64,
    synthesize_residual_view,
)

stage = str(CONFIG["execution"]["selected_stage"])
variant = str(CONFIG["execution"]["selected_variant"])
if stage not in CONFIG["execution"]["allowed_stages"]:
    raise ValueError(f"unsupported stage: {stage}")
if variant not in CONFIG["execution"]["allowed_selector_variants"]:
    raise ValueError(f"unsupported selector variant: {variant}")
RUN_SELECTOR = stage == "selector_train"
print(
    json.dumps(
        {
            "experiment": OUTPUT_PREFIX,
            "stage": stage,
            "variant": variant,
            "route": CONFIG["experiment"]["route"],
            "runtime": "CPU",
            "boosters_this_run": 20 if RUN_SELECTOR else 0,
            "parent_control_retraining": False,
            "validation_is_clean": True,
        },
        indent=2,
    )
)

# %% [markdown]
# ## 2. Fixed exp237 candidate surface

# %%
parent_config = exp237.load_config()
parent_config.setdefault("inference", {})["use_test_base_as_dense_auxiliary"] = False
candidates = exp237.candidate_specs_from_config(parent_config)
required = exp237.build_required_columns(parent_config, candidates)
frame, source_meta = exp237.load_train_feature_cache(
    cache_path=exp237.get_nested(parent_config, "data.exp099_train_feature_cache_local"),
    schema_path=exp237.get_nested(parent_config, "data.exp099_train_feature_schema_local"),
    required_columns=required,
    max_rows=None,
)
frame, enrichment_columns, enrichment_meta = exp237.add_feature_enrichment(
    frame, parent_config, max_rows=None
)
frame, cluster_columns, cluster_meta = exp237.add_cluster_prior_confidence_features(
    frame, parent_config, max_rows=None
)
frame, external_columns, external_meta = exp237.add_hmm_exp226_candidate_sources(
    frame, parent_config
)
frame, engineered_columns, _, _ = exp237.add_candidate_labels_and_features(
    frame, candidates, include_candidate_values=False
)
context_columns = exp237.select_numeric_feature_columns(
    frame,
    parent_config,
    [*engineered_columns, *enrichment_columns, *cluster_columns, *external_columns],
)
candidate_columns = [item.column for item in candidates]
candidate_names = [item.name for item in candidates]
configured_candidates = list(CONFIG["model"]["selector"]["candidates"])
if candidate_names != configured_candidates or len(candidate_names) != 11:
    raise ValueError(
        {"message": "candidate bank differs from exp238 runtime contract", "actual": candidate_names}
    )
print(
    {
        "rows": len(frame),
        "wells": int(frame.well.nunique()),
        "candidate_count": len(candidate_columns),
        "context_feature_count": len(context_columns),
        "source": source_meta,
    }
)

outer, inner = engine.deterministic_outer_inner_splits(
    frame,
    int(CONFIG["validation"]["outer_folds"]),
    int(CONFIG["validation"]["inner_folds"]),
)
fold_manifest_path = engine.save_fold_contract(OUTPUT_DIR, frame, outer, inner)
display(pd.read_csv(fold_manifest_path))

# %% [markdown]
# ## 3. Fold-safe residual profile audit

# %%
train_dir = exp237.ExperimentPaths().train_data_dir
if not train_dir.exists():
    raise FileNotFoundError(f"raw train directory does not exist: {train_dir}")
affine_cfg = CONFIG["augmentation"]["affine"]
profiles = {}
for well in sorted(frame["well"].astype(str).unique()):
    profiles[well] = read_residual_profile(
        well,
        train_dir / f"{well}__horizontal_well.csv",
        train_dir / f"{well}__typewell.csv",
        fit_scope=str(CONFIG["augmentation"]["fit_scope"]),
        trim_quantile=float(affine_cfg["trim_quantile"]),
        iterations=int(affine_cfg["iterations"]),
        min_points=int(affine_cfg["min_points"]),
        slope_bounds=tuple(float(value) for value in affine_cfg["slope_bounds"]),
    )

profile_table = pd.DataFrame([profile_inventory(profile) for profile in profiles.values()])
profile_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_residual_profile_inventory.csv"
profile_table.to_csv(profile_path, index=False)

fold_isolation = []
for outer_fold, ((outer_train, outer_valid), inner_splits) in enumerate(zip(outer, inner)):
    outer_valid_wells = set(frame.iloc[outer_valid].well.astype(str))
    for inner_fold, (inner_train, inner_valid) in enumerate(inner_splits):
        donor_wells = set(frame.iloc[inner_train].well.astype(str))
        inner_valid_wells = set(frame.iloc[inner_valid].well.astype(str))
        if donor_wells & inner_valid_wells or donor_wells & outer_valid_wells:
            raise AssertionError("validation well entered the residual donor pool")
        fold_isolation.append(
            {
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "donor_wells": len(donor_wells),
                "inner_valid_wells": len(inner_valid_wells),
                "outer_valid_wells": len(outer_valid_wells),
                "overlap": 0,
            }
        )

audit_wells = sorted(profiles)[:3]
audit_views = []
if len(audit_wells) >= 2:
    recipient = profiles[audit_wells[0]]
    donors = [profiles[well] for well in audit_wells[1:]]
    for audit_variant in CONFIG["execution"]["allowed_selector_variants"]:
        view = synthesize_residual_view(
            recipient,
            donors,
            variant=str(audit_variant),
            seed_parts=(CONFIG["reproducibility"]["seed"], "stage0", recipient.well),
            block_lengths=CONFIG["augmentation"]["block_lengths_rows"],
            rolling_window=int(CONFIG["augmentation"]["gr_rolling_window"]),
            residual_clip_abs=float(CONFIG["augmentation"]["residual_clip_abs"]),
        )
        audit_views.append(
            {
                "variant": audit_variant,
                "view_sha256": content_sha256(view.imputed_gr, view.missing_mask),
                "block_count": len(view.inventory),
                "block_lengths": [int(item["block_length"]) for item in view.inventory],
                "donor_wells": sorted(
                    {str(item["donor_well"]) for item in view.inventory}
                ),
                "block_contract_sha256": hashlib.sha256(
                    json.dumps(list(view.inventory), sort_keys=True).encode()
                ).hexdigest(),
                "missing_rows": int(view.missing_mask.sum()),
            }
        )

residual_audit = {
    "status": "residual_audit_complete",
    "rows": int(profile_table["rows"].sum()),
    "wells": len(profiles),
    "fit_scope": CONFIG["augmentation"]["fit_scope"],
    "profile_inventory_sha256": engine._sha(profile_path),
    "fold_isolation": fold_isolation,
    "negative_control_views": audit_views,
    "summary": {
        "median_fit_rmse": float(profile_table.fit_rmse.median()),
        "median_residual_std": float(profile_table.residual_std.median()),
        "median_missing_rate": float(profile_table.missing_rate.median()),
        "median_haar_dwt_detail_energy": float(profile_table.haar_dwt_detail_energy.median()),
        "median_fft_rotation_energy_ratio": float(
            profile_table.fft_rotation_energy_ratio.median()
        ),
    },
}
residual_audit_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_residual_audit_summary.json"
residual_audit_path.write_text(json.dumps(residual_audit, indent=2))
display(profile_table.describe(include="all"))

# %% [markdown]
# ## 4. Synthetic GR view and multi-observation feature rebuild
#
# Candidate TVT paths remain fixed. Only the target-free multi-observation GR
# likelihood columns in duplicated inner-train rows are regenerated.

# %%
def row_index_from_id(value: object) -> int:
    try:
        return int(str(value).rsplit("_", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"cannot parse row index from id={value}") from exc


def load_prefix_tvt(well: str, cache: dict[str, np.ndarray]) -> np.ndarray:
    if well not in cache:
        raw = pd.read_csv(train_dir / f"{well}__horizontal_well.csv", usecols=["TVT_input"])
        tvt = pd.to_numeric(raw["TVT_input"], errors="coerce")
        known = tvt.notna().to_numpy()
        if not known.any():
            raise ValueError(f"well {well} has no TVT_input prefix")
        stop = int(np.flatnonzero(known)[-1] + 1)
        cache[well] = (
            tvt.iloc[:stop]
            .interpolate(limit_direction="both")
            .ffill()
            .bfill()
            .to_numpy(np.float32)
        )
    return cache[well]


def nearest_indices(prefix_tvt: np.ndarray, values: np.ndarray) -> np.ndarray:
    order = np.argsort(prefix_tvt)
    sorted_tvt = prefix_tvt[order]
    positions = np.searchsorted(sorted_tvt, values, side="left")
    left = np.clip(positions - 1, 0, len(sorted_tvt) - 1)
    right = np.clip(positions, 0, len(sorted_tvt) - 1)
    choose_right = np.abs(sorted_tvt[right] - values) < np.abs(sorted_tvt[left] - values)
    return order[np.where(choose_right, right, left)].astype(np.int32)


def standardized(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=-1, keepdims=True)
    return centered / (values.std(axis=-1, keepdims=True) + 1e-6)


def multiobs_for_one_row(
    full_gr: np.ndarray,
    prefix_tvt: np.ndarray,
    row_index: int,
    values: np.ndarray,
    settings: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = np.asarray(settings["observation_offsets"], dtype=np.int32)
    nearest = nearest_indices(prefix_tvt, values)
    eval_index = np.clip(row_index + offsets, 0, len(full_gr) - 1)
    candidate_index = np.clip(nearest[:, None] + offsets[None, :], 0, len(full_gr) - 1)
    eval_vector = full_gr[eval_index][None, :]
    candidate_matrix = full_gr[candidate_index]
    mae = np.mean(np.abs(candidate_matrix - eval_vector), axis=1)
    ncc = np.mean(standardized(candidate_matrix) * standardized(eval_vector), axis=1)
    low, high = float(prefix_tvt.min()), float(prefix_tvt.max())
    range_distance = np.maximum(0.0, low - values) + np.maximum(0.0, values - high)
    range_penalty = np.exp(-range_distance / float(settings["out_of_range_scale"]))
    mae_score = np.exp(-mae / float(settings["gr_scale"]))
    ncc_score = np.clip((ncc + 1.0) / 2.0, 0.0, 1.0)
    score = np.clip(mae_score * (0.25 + 0.75 * ncc_score) * range_penalty, 0.0, 1.0)
    return score.astype(np.float32), mae.astype(np.float32), ncc.astype(np.float32)


def inventory_record(
    view,
    *,
    row_id: str,
    outer_fold: int,
    inner_fold: int,
    donor_wells: set[str],
) -> dict[str, Any]:
    blocks = [dict(item) for item in view.inventory]
    used = sorted({str(item["donor_well"]) for item in blocks})
    if not set(used) <= donor_wells and view.variant != "clean_duplicate":
        raise AssertionError("synthetic view used a donor outside inner-train wells")
    return {
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "row_id": row_id,
        "recipient_well": view.well,
        "variant": view.variant,
        "block_count": len(blocks),
        "block_lengths_json": json.dumps([item["block_length"] for item in blocks]),
        "donor_wells_json": json.dumps(used),
        "block_contract_sha256": hashlib.sha256(
            json.dumps(blocks, sort_keys=True).encode()
        ).hexdigest(),
        "view_sha256": content_sha256(view.imputed_gr, view.missing_mask),
        "missing_rows": int(view.missing_mask.sum()),
    }


def build_augmented_frame(
    base_rows: np.ndarray,
    donor_wells: set[str],
    *,
    outer_fold: int,
    inner_fold: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    augmented = frame.iloc[base_rows].copy().reset_index(drop=True)
    recompute_names = list(
        CONFIG["augmentation"]["multi_observation_likelihood"]["recompute_candidate_names"]
    )
    lookup = {item.name: item.column for item in candidates}
    recompute_columns = [lookup[name] for name in recompute_names]
    donor_profiles = [profiles[well] for well in sorted(donor_wells)]
    prefix_cache: dict[str, np.ndarray] = {}
    settings = CONFIG["augmentation"]["multi_observation_likelihood"]
    records = []
    for local_row, global_row in enumerate(base_rows):
        source = frame.iloc[int(global_row)]
        well = str(source.well)
        row_id = str(source.id)
        view = synthesize_residual_view(
            profiles[well],
            donor_profiles,
            variant=variant,
            seed_parts=(
                CONFIG["reproducibility"]["seed"],
                variant,
                outer_fold,
                inner_fold,
                well,
                row_id,
                0,
            ),
            block_lengths=CONFIG["augmentation"]["block_lengths_rows"],
            rolling_window=int(CONFIG["augmentation"]["gr_rolling_window"]),
            residual_clip_abs=float(CONFIG["augmentation"]["residual_clip_abs"]),
        )
        values = source[recompute_columns].to_numpy(dtype=np.float32)
        score, obs_mae, obs_ncc = multiobs_for_one_row(
            view.imputed_gr,
            load_prefix_tvt(well, prefix_cache),
            row_index_from_id(row_id),
            values,
            settings,
        )
        best = int(np.argmax(score))
        ordered = np.sort(score)
        generated: dict[str, float] = {
            "multiobs_top1": float(values[best]),
            "multiobs_score_max": float(score[best]),
            "multiobs_score_mean": float(score.mean()),
            "multiobs_score_gap": float(ordered[-1] - ordered[-2]),
            "multiobs_top1_source_id": float(best),
            "multiobs_top1_mae": float(obs_mae[best]),
            "multiobs_top1_ncc": float(obs_ncc[best]),
        }
        for index, name in enumerate(recompute_names):
            generated[f"multiobs_score_{name}"] = float(score[index])
            generated[f"multiobs_mae_{name}"] = float(obs_mae[index])
            generated[f"multiobs_ncc_{name}"] = float(obs_ncc[index])
        for temperature in settings["softmax_temperatures"]:
            temp = float(temperature)
            logits = score / max(temp, 1e-6)
            weights = np.exp(logits - logits.max())
            weights /= weights.sum()
            tag = str(temp).replace(".", "p")
            generated[f"multiobs_softmax_t{tag}"] = float(np.sum(values * weights))
        if "likpf_mean" in recompute_names:
            likpf = float(values[recompute_names.index("likpf_mean")])
            for weight in settings["likpf_blend_weights"]:
                alpha = float(weight)
                tag = str(alpha).replace(".", "p")
                generated[f"likpf_multiobs_blend_w{tag}"] = (
                    (1.0 - alpha) * likpf + alpha * generated["multiobs_top1"]
                )
        for column, value in generated.items():
            if column in augmented.columns:
                augmented.at[local_row, column] = np.float32(value)
        records.append(
            inventory_record(
                view,
                row_id=row_id,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                donor_wells=donor_wells,
            )
        )
    return augmented, records

# %% [markdown]
# ## 5. Strict nested selector training

# %%
def fit_augmented_nested_selector():
    selector_cfg = CONFIG["model"]["selector"]
    seed = int(CONFIG["reproducibility"]["seed"])
    nested_outputs = []
    model_manifest = []
    augmentation_inventory = []
    model_dir = OUTPUT_DIR / f"{OUTPUT_PREFIX}_selector_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for outer_fold, ((outer_train, outer_valid), inner_splits) in enumerate(zip(outer, inner)):
        train_scores = np.full((len(frame), len(candidate_columns)), np.nan, np.float32)
        valid_models = []
        outer_valid_wells = set(frame.iloc[outer_valid].well.astype(str))
        for inner_fold, (train_rows, valid_rows) in enumerate(inner_splits):
            donor_wells = set(frame.iloc[train_rows].well.astype(str))
            valid_wells = set(frame.iloc[valid_rows].well.astype(str))
            if donor_wells & valid_wells or donor_wells & outer_valid_wells:
                raise AssertionError("nested donor fold isolation failed")
            fit_train_rows = engine._bounded_base_rows(
                train_rows,
                len(candidate_columns),
                int(selector_cfg["max_train_long_rows_per_model"]),
                seed + 10_000 * outer_fold + 100 * inner_fold,
            )
            fit_valid_rows = engine._bounded_base_rows(
                valid_rows,
                len(candidate_columns),
                int(selector_cfg["max_valid_long_rows_per_model"]),
                seed + 20_000 * outer_fold + 100 * inner_fold,
            )
            fraction = float(CONFIG["augmentation"]["augmented_base_row_fraction"])
            max_augmented = int(CONFIG["augmentation"]["max_augmented_base_rows_per_model"])
            augmented_count = min(
                max_augmented, max(1, int(round(len(fit_train_rows) * fraction)))
            )
            rng = np.random.default_rng(
                stable_uint64(seed, variant, outer_fold, inner_fold, "recipient_rows")
            )
            augmented_rows = np.sort(
                rng.choice(fit_train_rows, size=augmented_count, replace=False)
            )
            augmented_frame, records = build_augmented_frame(
                augmented_rows,
                donor_wells,
                outer_fold=outer_fold,
                inner_fold=inner_fold,
            )
            augmentation_inventory.extend(records)
            x_clean, y_clean = engine.candidate_long(
                frame,
                fit_train_rows,
                candidate_columns,
                context_columns,
                with_target=True,
            )
            x_augmented, y_augmented = engine.candidate_long(
                augmented_frame,
                np.arange(len(augmented_frame)),
                candidate_columns,
                context_columns,
                with_target=True,
            )
            x_train = pd.concat([x_clean, x_augmented], ignore_index=True)
            y_train = np.concatenate([y_clean, y_augmented])
            x_valid, y_valid = engine.candidate_long(
                frame,
                fit_valid_rows,
                candidate_columns,
                context_columns,
                with_target=True,
            )
            model = lgb.LGBMRegressor(
                objective="regression_l1",
                random_state=seed + 100 * outer_fold + inner_fold,
                **dict(selector_cfg["params"]),
            )
            model.fit(
                x_train,
                y_train,
                eval_set=[(x_valid, y_valid)],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
            )
            model_path = model_dir / f"selector_outer{outer_fold}_inner{inner_fold}.txt"
            model.booster_.save_model(str(model_path), num_iteration=model.best_iteration_)
            train_scores[valid_rows] = engine.predict_candidate_errors(
                model,
                frame,
                valid_rows,
                candidate_columns,
                context_columns,
                chunk_rows=int(selector_cfg["predict_chunk_rows"]),
            )
            valid_models.append(model)
            model_manifest.append(
                {
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "variant": variant,
                    "clean_train_base_rows": len(fit_train_rows),
                    "augmented_train_base_rows": len(augmented_rows),
                    "clean_valid_base_rows": len(fit_valid_rows),
                    "train_long_rows": len(x_train),
                    "valid_long_rows": len(x_valid),
                    "donor_wells": len(donor_wells),
                    "validation_augmented": False,
                    "best_iteration": int(model.best_iteration_),
                    "file": str(model_path),
                    "sha256": engine._sha(model_path),
                    "feature_count": int(model.booster_.num_feature()),
                }
            )
            del augmented_frame, x_clean, x_augmented, x_train, x_valid
            del y_clean, y_augmented, y_train, y_valid
            gc.collect()
        if not np.isfinite(train_scores[outer_train]).all():
            raise AssertionError(f"outer fold {outer_fold}: incomplete clean inner OOF scores")
        valid_scores = np.mean(
            [
                engine.predict_candidate_errors(
                    model,
                    frame,
                    outer_valid,
                    candidate_columns,
                    context_columns,
                    chunk_rows=int(selector_cfg["predict_chunk_rows"]),
                )
                for model in valid_models
            ],
            axis=0,
        ).astype(np.float32)
        nested_outputs.append(
            {
                "outer_train": outer_train,
                "outer_valid": outer_valid,
                "train_scores": train_scores[outer_train],
                "valid_scores": valid_scores,
            }
        )
        del valid_models
        gc.collect()
    return nested_outputs, model_manifest, augmentation_inventory


if RUN_SELECTOR:
    nested, model_manifest, augmentation_inventory = fit_augmented_nested_selector()
else:
    nested, model_manifest, augmentation_inventory = [], [], []
    print("Stage 0 only: selector fitting is intentionally skipped.")

# %% [markdown]
# ## 6. Historical calibration and safety guard

# %%
def find_historical_selector_dir() -> Path:
    prefix = "exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218"
    configured = Path(CONFIG["data"]["historical_selector_artifact_dir_local"])
    candidates_dir = [
        configured,
        Path("/kaggle/input/exp238-nested-selector-train/artifacts"),
        Path("/kaggle/input/exp238-nested-selector-train"),
        Path("/kaggle/input/notebooks/kentookumura/exp238-nested-selector-train/artifacts"),
        Path("/kaggle/input/notebooks/kentookumura/exp238-nested-selector-train"),
    ]
    resolved = next(
        (
            path
            for path in candidates_dir
            if (path / f"{prefix}_nested_scores_outer0.csv.gz").exists()
        ),
        None,
    )
    if resolved is None and Path("/kaggle/input").exists():
        matches = list(Path("/kaggle/input").rglob(f"{prefix}_nested_scores_outer0.csv.gz"))
        resolved = matches[0].parent if matches else None
    if resolved is None:
        raise FileNotFoundError(
            "historical exp238 nested scores are required for selector_train guard"
        )
    return resolved


def load_historical_scores(directory: Path) -> list[dict[str, np.ndarray]]:
    prefix = "exp238_nested_hmm_exp226_selector_rank_slot_addonly_on_exp218"
    score_columns = [f"pred_error__{column}" for column in candidate_columns]
    outputs = []
    for outer_fold in range(len(outer)):
        artifact = pd.read_csv(
            directory / f"{prefix}_nested_scores_outer{outer_fold}.csv.gz",
            dtype={"id": str, "well": str},
        ).sort_values("row_index")
        if not artifact[engine.KEYS].reset_index(drop=True).equals(
            frame[engine.KEYS].astype(str).reset_index(drop=True)
        ):
            raise ValueError(f"historical exp238 row alignment failed for fold {outer_fold}")
        valid = artifact.loc[artifact.role.eq("valid")].sort_values("row_index")
        valid_rows = valid.row_index.to_numpy(np.int64)
        if not np.array_equal(valid_rows, np.sort(outer[outer_fold][1])):
            raise ValueError(f"historical fold contract differs for outer {outer_fold}")
        outputs.append(
            {
                "outer_valid": valid_rows,
                "valid_scores": valid[score_columns].to_numpy(np.float32),
            }
        )
    return outputs


def selector_readout(
    source: str, score_items: list[dict[str, np.ndarray]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calibration_rows = []
    safety_parts = []
    by_well_parts = []
    candidate_values = frame[candidate_columns].to_numpy(np.float32)
    true_tvt = (
        frame.last_known_tvt.to_numpy(np.float32) + frame.target.to_numpy(np.float32)
    )
    for outer_fold, item in enumerate(score_items):
        rows = item["outer_valid"]
        scores = item["valid_scores"]
        absolute_error = np.abs(candidate_values[rows] - true_tvt[rows, None])
        within = (absolute_error <= 10.0).reshape(-1)
        auc = float(roc_auc_score(within, -scores.reshape(-1))) if np.unique(within).size > 1 else np.nan
        sorted_scores = np.sort(scores, axis=1)
        calibration_rows.append(
            {
                "source": source,
                "outer_fold": outer_fold,
                "rows": len(rows),
                "candidate_auc_within10": auc,
                "expected_error_mae": float(np.mean(np.abs(scores - absolute_error))),
                "rank_margin_mean": float(np.mean(sorted_scores[:, 1] - sorted_scores[:, 0])),
                "top1_accuracy": float(
                    np.mean(np.argmin(scores, axis=1) == np.argmin(absolute_error, axis=1))
                ),
            }
        )
        safety, by_well = engine.selector_safety_readout(
            frame, rows, scores, candidate_columns, "likpf_mean"
        )
        safety["source"] = source
        safety["outer_fold"] = outer_fold
        by_well["source"] = source
        by_well["outer_fold"] = outer_fold
        safety_parts.append(safety)
        by_well_parts.append(by_well)
    return (
        pd.DataFrame(calibration_rows),
        pd.concat(safety_parts, ignore_index=True),
        pd.concat(by_well_parts, ignore_index=True),
    )


if RUN_SELECTOR:
    historical_dir = find_historical_selector_dir()
    historical_nested = load_historical_scores(historical_dir)
    current_calibration, current_safety, current_by_well = selector_readout(variant, nested)
    historical_calibration, historical_safety, historical_by_well = selector_readout(
        "historical_exp238", historical_nested
    )
    calibration = pd.concat([historical_calibration, current_calibration], ignore_index=True)
    safety = pd.concat([historical_safety, current_safety], ignore_index=True)
    by_well = pd.concat([historical_by_well, current_by_well], ignore_index=True)

    def mean_safety(source: str, bucket: str) -> float:
        return float(
            safety.loc[safety.source.eq(source) & safety.bucket.eq(bucket), "delta_rmse"].mean()
        )

    historical_worst = float(
        by_well.loc[by_well.source.eq("historical_exp238"), "delta_rmse"].max()
    )
    current_worst = float(by_well.loc[by_well.source.eq(variant), "delta_rmse"].max())
    historical_expected_mae = float(historical_calibration.expected_error_mae.mean())
    current_expected_mae = float(current_calibration.expected_error_mae.mean())
    historical_auc = float(historical_calibration.candidate_auc_within10.mean())
    current_auc = float(current_calibration.candidate_auc_within10.mean())
    guard_cfg = CONFIG["validation"]["selector_guard"]
    checks = {
        "global_nonworse": mean_safety(variant, "global")
        <= mean_safety("historical_exp238", "global")
        + float(guard_cfg["max_global_delta_vs_historical"]),
        "near_nonworse": mean_safety(variant, "000_050")
        <= mean_safety("historical_exp238", "000_050")
        + float(guard_cfg["max_near_delta_vs_historical"]),
        "longtail_nonworse": mean_safety(variant, "1000_plus")
        <= mean_safety("historical_exp238", "1000_plus")
        + float(guard_cfg["max_longtail_delta_vs_historical"]),
        "expected_error_mae_nonworse": current_expected_mae
        <= historical_expected_mae
        + float(guard_cfg["max_expected_error_mae_delta_vs_historical"]),
        "candidate_auc_nonworse": current_auc
        >= historical_auc - float(guard_cfg["max_candidate_auc_drop_vs_historical"]),
        "worst_well_nonworse": current_worst
        <= historical_worst
        + float(guard_cfg["max_worst_well_regression_delta_vs_historical"]),
    }
    guard_pass = bool(all(checks.values()))
    decision = {
        "guard_pass": guard_pass,
        "checks": checks,
        "current": {
            "global_delta_rmse_vs_likpf": mean_safety(variant, "global"),
            "near_delta_rmse_vs_likpf": mean_safety(variant, "000_050"),
            "longtail_delta_rmse_vs_likpf": mean_safety(variant, "1000_plus"),
            "worst_well_delta_rmse_vs_likpf": current_worst,
            "expected_error_mae": current_expected_mae,
            "candidate_auc_within10": current_auc,
        },
        "historical_exp238": {
            "global_delta_rmse_vs_likpf": mean_safety("historical_exp238", "global"),
            "near_delta_rmse_vs_likpf": mean_safety("historical_exp238", "000_050"),
            "longtail_delta_rmse_vs_likpf": mean_safety("historical_exp238", "1000_plus"),
            "worst_well_delta_rmse_vs_likpf": historical_worst,
            "expected_error_mae": historical_expected_mae,
            "candidate_auc_within10": historical_auc,
        },
    }
    print(json.dumps(decision, indent=2))
else:
    calibration = pd.DataFrame()
    safety = pd.DataFrame()
    by_well = pd.DataFrame()
    decision = {"guard_pass": False, "reason": "selector_train_not_executed"}

# %% [markdown]
# ## 7. Artifacts and decision

# %%
if RUN_SELECTOR:
    score_manifest = engine.save_nested_score_artifacts(
        OUTPUT_DIR, frame, nested, candidate_columns
    )
    inventory_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_augmentation_inventory.csv.gz"
    model_manifest_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_selector_model_manifest.csv"
    calibration_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_selector_calibration.csv"
    safety_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_selector_safety_metrics.csv"
    by_well_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_selector_by_well.csv"
    pd.DataFrame(augmentation_inventory).to_csv(
        inventory_path, index=False, compression="gzip"
    )
    pd.DataFrame(model_manifest).to_csv(model_manifest_path, index=False)
    calibration.to_csv(calibration_path, index=False)
    safety.to_csv(safety_path, index=False)
    by_well.to_csv(by_well_path, index=False)
    summary = {
        "status": (
            "selector_guard_passed_final_train_allowed"
            if decision["guard_pass"]
            else "selector_guard_failed_final_train_forbidden"
        ),
        "stage": stage,
        "variant": variant,
        "rows": len(frame),
        "wells": int(frame.well.nunique()),
        "candidate_columns": candidate_columns,
        "candidate_names": candidate_names,
        "context_feature_count": len(context_columns),
        "selector_model_count": len(model_manifest),
        "validation_augmented": False,
        "decision": decision,
        "score_artifacts": score_manifest,
        "sha256": {
            "residual_profile_inventory": engine._sha(profile_path),
            "residual_audit": engine._sha(residual_audit_path),
            "fold_manifest": engine._sha(fold_manifest_path),
            "augmentation_inventory_decompressed": engine._sha(
                inventory_path, decompressed=True
            ),
            "selector_model_manifest": engine._sha(model_manifest_path),
            "calibration": engine._sha(calibration_path),
            "safety": engine._sha(safety_path),
        },
    }
    summary_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_selector_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    display(calibration)
    display(safety)
    display(by_well.sort_values("delta_rmse", ascending=False).head(20))
else:
    print(json.dumps(residual_audit, indent=2))
