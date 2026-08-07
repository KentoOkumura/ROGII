# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp259 coordinate equivariance / path warp augmentation — train audit
#
# This notebook is the authoritative Kaggle entrypoint for the transform-contract
# audit. It does not fit a model. Exact coordinate symmetries are inverted and
# compared with their source well. Approximate path warps keep the official prefix
# fixed, resample evaluation-tail GR from the same typewell, regenerate local and
# spectral diagnostics, and are rejected when they leave the real-train envelope.

# %% [markdown]
# ## Contents
# 1. Imports and runtime helpers
# 2. Experiment and compute contract
# 3. Raw train input checks
# 4. Clean-well distribution envelope
# 5. Exact and approximate transform audit
# 6. Metrics and generated artifacts
# 7. SHA and adoption guard

# %% [markdown]
# ## 1. Imports and runtime helpers

# %%
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from IPython.display import display

EXPERIMENT_NAME = "exp259_coordinate_equivariance_path_warp_augmentation"
OUTPUT_PREFIX = EXPERIMENT_NAME


def find_package_dir() -> Path:
    cwd = Path.cwd()
    candidates = [cwd, *cwd.parents]
    for candidate in candidates:
        direct = candidate / "config.yaml"
        if direct.exists() and EXPERIMENT_NAME in direct.read_text():
            return candidate
        nested = candidate / "experiments" / EXPERIMENT_NAME
        if (nested / "config.yaml").exists():
            return nested
    raise FileNotFoundError(f"could not locate {EXPERIMENT_NAME}/config.yaml from {cwd}")


PACKAGE_DIR = find_package_dir()
REPO_ROOT = PACKAGE_DIR.parents[1] if PACKAGE_DIR.parent.name == "experiments" else PACKAGE_DIR
for candidate in (REPO_ROOT, PACKAGE_DIR, Path.cwd()):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from src.coordinate_path_augmentation import (  # noqa: E402
    ALL_TRANSFORMS,
    STRICT_TRANSFORMS,
    apply_transform,
    choose_transform_spec,
    evaluate_distribution_guard,
    exact_inverse_error,
    fit_distribution_envelope,
    inverse_exact_transform,
    parameter_manifest,
    read_horizontal,
    read_typewell,
    sha256_file,
    sha256_gzip_decompressed,
    summarize_well,
)


def get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    value: Any = config
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def resolve_train_dir(configured: str) -> Path:
    name = Path(configured).name
    candidates = [
        Path(configured),
        REPO_ROOT / configured,
        PACKAGE_DIR / configured,
        Path("/kaggle/input/rogii-wellbore-geology-prediction") / name,
        Path("/kaggle/input/rogii-wellbore-geology-prediction/data/raw") / name,
    ]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(path for path in kaggle_input.glob("**/train") if path.is_dir())
    for candidate in candidates:
        if candidate.exists() and next(candidate.glob("*__horizontal_well.csv"), None) is not None:
            return candidate.resolve()
    checked = "\n".join(str(candidate) for candidate in candidates[:40])
    raise FileNotFoundError(f"could not resolve raw train directory; checked:\n{checked}")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=json_default) + "\n")


def json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def exact_metric_delta(clean: dict[str, Any], transformed: dict[str, Any]) -> float:
    keys = [
        "md_step_q01",
        "md_step_q99",
        "xy_slope_q95",
        "z_slope_abs_q95",
        "xy_curvature_q95",
        "z_curvature_abs_q95",
        "tvt_slope_abs_q95",
        "typewell_coverage_fraction",
        "gr_fft_low_fraction",
        "gr_fft_mid_fraction",
        "gr_fft_high_fraction",
        "gr_haar_level1_energy",
        "gr_haar_level2_energy",
        "gr_haar_level3_energy",
    ]
    deltas = []
    for key in keys:
        left = float(clean[key])
        right = float(transformed[key])
        if np.isfinite(left) and np.isfinite(right):
            scale = max(abs(left), abs(right), 1.0)
            deltas.append(abs(left - right) / scale)
    return float(max(deltas, default=0.0))


# %% [markdown]
# ## 2. Experiment and compute contract

# %%
CONFIG_PATH = PACKAGE_DIR / "config.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text()) or {}
assert get_nested(CONFIG, "experiment.name") == EXPERIMENT_NAME
assert get_nested(CONFIG, "experiment.route") == "ensemble"
assert get_nested(CONFIG, "execution.stage") == "transform_audit_only"
assert get_nested(CONFIG, "model.trains_new_boosters") is False
assert get_nested(CONFIG, "model.planned_boosters") == 0
assert get_nested(CONFIG, "submission.enabled") is False

SEED = int(get_nested(CONFIG, "validation.seed", 42))
MAX_WELLS = get_nested(CONFIG, "execution.max_wells")
MAX_WELLS = None if MAX_WELLS is None else int(MAX_WELLS)
PREVIEW_WELLS = int(get_nested(CONFIG, "execution.preview_wells", 3))
VIEW_SLOTS = int(get_nested(CONFIG, "execution.view_slots_per_transform", 1))
EXACT_KINDS = list(get_nested(CONFIG, "augmentation.exact_transforms", []))
APPROXIMATE_KINDS = list(get_nested(CONFIG, "augmentation.approximate_transforms", []))
TRANSFORM_KINDS = [*EXACT_KINDS, *APPROXIMATE_KINDS]
assert set(TRANSFORM_KINDS) == set(ALL_TRANSFORMS)
assert len(TRANSFORM_KINDS) == len(set(TRANSFORM_KINDS))
assert get_nested(CONFIG, "augmentation.compose_transforms") is False

print(
    json.dumps(
        {
            "experiment": EXPERIMENT_NAME,
            "route": get_nested(CONFIG, "experiment.route"),
            "stage": get_nested(CONFIG, "execution.stage"),
            "parent": get_nested(CONFIG, "lineage.parent"),
            "transforms": TRANSFORM_KINDS,
            "view_slots": VIEW_SLOTS,
            "seed": SEED,
            "active_variants": 0,
            "model_configs": 0,
            "folds": 0,
            "boosters": 0,
            "parent_control_retraining": False,
            "gpu": get_nested(CONFIG, "runtime.kaggle.enable_gpu"),
            "internet": get_nested(CONFIG, "runtime.kaggle.enable_internet"),
        },
        indent=2,
    )
)


# %% [markdown]
# ## 3. Raw train input checks

# %%
TRAIN_DIR = resolve_train_dir(str(get_nested(CONFIG, "data.train_dir", "data/raw/train")))
OUTPUT_DIR = PACKAGE_DIR / str(get_nested(CONFIG, "outputs.directory", "artifacts"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

horizontal_suffix = str(get_nested(CONFIG, "data.horizontal_suffix", "__horizontal_well.csv"))
typewell_suffix = str(get_nested(CONFIG, "data.typewell_suffix", "__typewell.csv"))
horizontal_paths = sorted(TRAIN_DIR.glob(f"*{horizontal_suffix}"))
well_ids = [path.name[: -len(horizontal_suffix)] for path in horizontal_paths]
if MAX_WELLS is not None:
    well_ids = well_ids[:MAX_WELLS]
if not well_ids:
    raise ValueError(f"no horizontal train wells found in {TRAIN_DIR}")

missing_typewells = [
    well for well in well_ids if not (TRAIN_DIR / f"{well}{typewell_suffix}").exists()
]
if missing_typewells:
    raise FileNotFoundError(f"missing typewell files for {missing_typewells[:10]}")

print({"train_dir": str(TRAIN_DIR), "wells": len(well_ids), "first_wells": well_ids[:5]})


# %% [markdown]
# ## 4. Clean-well distribution envelope

# %%
clean_summaries: list[dict[str, Any]] = []
input_sha: dict[str, dict[str, str]] = {}
for well in well_ids:
    horizontal_path = TRAIN_DIR / f"{well}{horizontal_suffix}"
    typewell_path = TRAIN_DIR / f"{well}{typewell_suffix}"
    horizontal = read_horizontal(horizontal_path, require_target=True)
    typewell = read_typewell(typewell_path)
    clean_summaries.append({"well": well, **summarize_well(horizontal, typewell)})
    input_sha[well] = {
        "horizontal_sha256": sha256_file(horizontal_path),
        "typewell_sha256": sha256_file(typewell_path),
    }

clean_frame = pd.DataFrame(clean_summaries).sort_values("well").reset_index(drop=True)
guard_config = dict(get_nested(CONFIG, "augmentation.guard", {}))
envelope = fit_distribution_envelope(
    clean_frame,
    lower_quantile=float(guard_config["lower_quantile"]),
    upper_quantile=float(guard_config["upper_quantile"]),
    relative_margin=float(guard_config["relative_margin"]),
    min_typewell_coverage=float(guard_config["min_typewell_coverage"]),
)

real_summary_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_real_well_summary.csv"
envelope_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_distribution_envelope.json"
clean_frame.to_csv(real_summary_path, index=False)
write_json(envelope_path, envelope)
display(clean_frame.head())
display(pd.DataFrame(envelope["metrics"]).T)


# %% [markdown]
# ## 5. Exact and approximate transform audit

# %%
parameter_grid = dict(get_nested(CONFIG, "augmentation.parameter_grid", {}))
inverse_tolerance = float(guard_config["inverse_tolerance"])
anchor_tolerance = float(guard_config["anchor_tolerance"])
manifest_rows: list[dict[str, Any]] = []
preview_rows: list[dict[str, Any]] = []

clean_lookup = clean_frame.set_index("well").to_dict(orient="index")
for well_index, well in enumerate(well_ids):
    horizontal_path = TRAIN_DIR / f"{well}{horizontal_suffix}"
    typewell_path = TRAIN_DIR / f"{well}{typewell_suffix}"
    horizontal = read_horizontal(horizontal_path, require_target=True)
    typewell = read_typewell(typewell_path)
    clean_summary = clean_lookup[well]

    for transform_kind in TRANSFORM_KINDS:
        for view_slot in range(VIEW_SLOTS):
            spec = choose_transform_spec(
                transform_kind,
                seed=SEED,
                well=well,
                view_slot=view_slot,
                parameter_grid=dict(parameter_grid.get(transform_kind) or {}),
            )
            result = apply_transform(horizontal, typewell, spec)
            transformed_summary = summarize_well(
                result.horizontal, result.typewell, metadata=result.metadata
            )

            inverse_errors: dict[str, float] = {}
            inverse_max_abs: float | None = None
            invariant_metric_delta: float | None = None
            if spec.exact:
                inverted = inverse_exact_transform(result, spec)
                inverse_errors = exact_inverse_error(horizontal, typewell, inverted)
                inverse_max_abs = float(inverse_errors["max_abs"])
                invariant_metric_delta = exact_metric_delta(clean_summary, transformed_summary)

            accepted, reject_reasons = evaluate_distribution_guard(
                transformed_summary,
                envelope,
                exact=spec.exact,
                inverse_max_abs=inverse_max_abs,
                inverse_tolerance=inverse_tolerance,
                anchor_tolerance=anchor_tolerance,
            )
            if (
                spec.exact
                and invariant_metric_delta is not None
                and invariant_metric_delta > inverse_tolerance
            ):
                accepted = False
                reject_reasons.append("exact_local_metric_invariance")

            manifest_rows.append(
                {
                    "well": well,
                    **parameter_manifest(spec),
                    **input_sha[well],
                    **transformed_summary,
                    "inverse_max_abs": inverse_max_abs,
                    "exact_local_metric_relative_delta_max": invariant_metric_delta,
                    "accepted": bool(accepted),
                    "reject_reasons": "|".join(sorted(set(reject_reasons))),
                    "inverse_error_json": json.dumps(inverse_errors, sort_keys=True),
                }
            )

            if well_index < PREVIEW_WELLS:
                anchor_index = int(result.metadata["anchor_index"])
                for row_index in sorted(
                    {
                        0,
                        anchor_index,
                        min(anchor_index + 1, len(horizontal) - 1),
                        len(horizontal) - 1,
                    }
                ):
                    original_row = horizontal.iloc[row_index]
                    transformed_row = result.horizontal.iloc[row_index]
                    preview_rows.append(
                        {
                            "well": well,
                            "transform_kind": transform_kind,
                            "view_slot": view_slot,
                            "row_index": row_index,
                            "is_anchor": row_index == anchor_index,
                            "original_MD": original_row["MD"],
                            "transformed_MD": transformed_row["MD"],
                            "original_X": original_row["X"],
                            "transformed_X": transformed_row["X"],
                            "original_Y": original_row["Y"],
                            "transformed_Y": transformed_row["Y"],
                            "original_Z": original_row["Z"],
                            "transformed_Z": transformed_row["Z"],
                            "original_TVT": original_row["TVT"],
                            "transformed_TVT": transformed_row["TVT"],
                            "original_GR": original_row["GR"],
                            "transformed_GR": transformed_row["GR"],
                        }
                    )

manifest = (
    pd.DataFrame(manifest_rows)
    .sort_values(["well", "transform_kind", "view_slot"])
    .reset_index(drop=True)
)
preview = (
    pd.DataFrame(preview_rows)
    .sort_values(["well", "transform_kind", "view_slot", "row_index"])
    .reset_index(drop=True)
)


# %% [markdown]
# ## 6. Metrics and generated artifacts

# %%
transform_summary = (
    manifest.groupby(["transform_class", "transform_kind"], as_index=False)
    .agg(
        views=("well", "size"),
        wells=("well", "nunique"),
        accepted=("accepted", "sum"),
        accept_rate=("accepted", "mean"),
        inverse_max_abs=("inverse_max_abs", "max"),
        exact_metric_delta_max=("exact_local_metric_relative_delta_max", "max"),
        anchor_max_abs_delta=("anchor_max_abs_delta", "max"),
        typewell_coverage_min=("typewell_coverage_fraction", "min"),
        xy_slope_q95_median=("xy_slope_q95", "median"),
        tvt_slope_abs_q95_median=("tvt_slope_abs_q95", "median"),
    )
    .sort_values(["transform_class", "transform_kind"])
    .reset_index(drop=True)
)

manifest_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_transform_manifest.csv.gz"
transform_summary_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_transform_summary.csv"
preview_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_preview_rows.csv.gz"
manifest.to_csv(
    manifest_path,
    index=False,
    compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
)
transform_summary.to_csv(transform_summary_path, index=False)
preview.to_csv(
    preview_path,
    index=False,
    compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
)
display(transform_summary)
display(manifest.loc[~manifest["accepted"], ["well", "transform_kind", "reject_reasons"]].head(30))


# %% [markdown]
# ## 7. SHA and adoption guard

# %%
exact_manifest = manifest[manifest["transform_kind"].isin(STRICT_TRANSFORMS)]
approx_manifest = manifest[~manifest["transform_kind"].isin(STRICT_TRANSFORMS)]
exact_pass = bool(exact_manifest["accepted"].all())
approx_accept_rate_by_transform = {
    str(row.transform_kind): float(row.accept_rate)
    for row in transform_summary.itertuples()
    if row.transform_class == "approximate"
}

summary = {
    "experiment": EXPERIMENT_NAME,
    "status": "transform_audit_completed" if exact_pass else "transform_audit_guard_failed",
    "stage": "transform_audit_only",
    "route": "ensemble",
    "seed": SEED,
    "wells": int(len(well_ids)),
    "transforms": int(len(TRANSFORM_KINDS)),
    "view_slots_per_transform": VIEW_SLOTS,
    "views": int(len(manifest)),
    "active_variants": 0,
    "model_configs": 0,
    "folds": 0,
    "boosters": 0,
    "parent_control_retraining": False,
    "exact_inverse_guard_pass": exact_pass,
    "exact_inverse_max_abs": float(exact_manifest["inverse_max_abs"].max()),
    "exact_local_metric_relative_delta_max": float(
        exact_manifest["exact_local_metric_relative_delta_max"].max()
    ),
    "approximate_accept_rate_by_transform": approx_accept_rate_by_transform,
    "accepted_views": int(manifest["accepted"].sum()),
    "rejected_views": int((~manifest["accepted"]).sum()),
    "training_allowed": False,
    "inference_allowed": False,
    "submission_allowed": False,
    "artifacts": {
        "real_well_summary": {
            "path": str(real_summary_path),
            "sha256": sha256_file(real_summary_path),
        },
        "distribution_envelope": {
            "path": str(envelope_path),
            "sha256": sha256_file(envelope_path),
            "content_sha256": envelope["content_sha256"],
        },
        "transform_manifest": {
            "path": str(manifest_path),
            "raw_sha256": sha256_file(manifest_path),
            "decompressed_content_sha256": sha256_gzip_decompressed(manifest_path),
        },
        "transform_summary": {
            "path": str(transform_summary_path),
            "sha256": sha256_file(transform_summary_path),
        },
        "preview_rows": {
            "path": str(preview_path),
            "raw_sha256": sha256_file(preview_path),
            "decompressed_content_sha256": sha256_gzip_decompressed(preview_path),
        },
        "config": {"path": str(CONFIG_PATH), "sha256": sha256_file(CONFIG_PATH)},
    },
    "next_action": (
        "review transform acceptance and request an explicit clean-control/"
        "augmentation training compute contract"
        if exact_pass
        else "fix exact inverse or local-metric consistency before any augmentation training"
    ),
}
summary_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_summary.json"
write_json(summary_path, summary)
print(json.dumps(summary, indent=2, default=json_default))

if not exact_pass:
    raise RuntimeError(
        "exact coordinate-equivariance guard failed; model training remains forbidden"
    )
