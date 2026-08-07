# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # exp263 last-anchor-better candidate cache — hidden-safe inference
#
# Stage 0で固定したdeployability/formula契約に従い、raw competition testから6 primitiveを
# 再生成する。5 fixed pairと`exp226_w500_50_50`を再構成し、固定formulaだけを
# `submission.csv`へ出力する。train-only候補、fold別fit、selector outputは使用しない。

# %% [markdown]
# ## Contents
#
# 1. Imports
# 2. Runtime, source, and identity helpers
# 3. Setup and submission contract
# 4. Trusted upstream source resolution
# 5. Raw-test six-primitive regeneration
# 6. Five pair and fixed named-formula parity
# 7. Current-test reference parity and submission generation
# 8. SHA, metrics, and generated artifacts

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import json
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from candidate_cache_builder import (
    assemble_stage1_current_test_parity,
    attach_stage1_current_test_confidence,
    build_submission_from_stage1_parity,
)
from candidate_cache_contract import (
    NAMED_COMBINATIONS,
    PAIR_SHORTLIST,
    RAWTEST_CORE_CANDIDATE_IDS,
    STAGE1_NATIVE_CONFIDENCE_FIELDS,
    validate_contract,
)
from candidate_cache_loader import frame_content_sha256, sha256_file
from IPython.display import display
from settings import EXPERIMENT_NAME, ExperimentPaths, get_nested, load_config

# %% [markdown]
# ## 2. Runtime, source, and identity helpers

# %%
KAGGLE_INPUT_ROOT = Path("/kaggle/input")


def resolve_unique_source(filename: str, path_token: str) -> Path:
    matches = [
        path
        for path in sorted(KAGGLE_INPUT_ROOT.rglob(filename))
        if path_token in str(path)
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one {filename} under source token {path_token}, got {matches}"
        )
    return matches[0]


def copy_trusted_source(source: Path, target_dir: Path, module_name: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{module_name}.py"
    shutil.copy2(source, target)
    return target


def parse_identity(frame: pd.DataFrame) -> pd.DataFrame:
    ids = frame["id"].astype(str)
    split = ids.str.rsplit("_", n=1, expand=True)
    if split.shape[1] != 2:
        raise ValueError("candidate id must use <well>_<row_idx>")
    return pd.DataFrame(
        {
            "id": ids,
            "well": split[0].astype(str),
            "well_row_idx": pd.to_numeric(split[1], errors="raise").astype(np.int32),
        }
    )


def finalize_primitive_confidence(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    excluded = {"id", "well", "well_row_idx", "candidate_tvt", "confidence_valid"}
    native_fields = [column for column in output.columns if column not in excluded]
    available: list[np.ndarray] = []
    for field in native_fields:
        values = pd.to_numeric(output[field], errors="coerce").to_numpy(np.float32)
        output[field] = values
        available.append(np.isfinite(values))
    candidate_finite = np.isfinite(output["candidate_tvt"].to_numpy(np.float32))
    output["confidence_valid"] = (
        candidate_finite & np.logical_or.reduce(available)
        if available
        else np.zeros(len(output), dtype=bool)
    )
    return output


def standard_primitive(
    frame: pd.DataFrame,
    value: Any,
    *,
    confidence: dict[str, Any] | None = None,
) -> pd.DataFrame:
    output = parse_identity(frame)
    output["candidate_tvt"] = np.asarray(value, dtype=np.float32)
    for field, field_value in (confidence or {}).items():
        output[field] = np.asarray(field_value, dtype=np.float32)
    return finalize_primitive_confidence(output)


def generate_hmm_primitive(
    *,
    list_well_ids: Callable[[str | Path], list[str]],
    load_well: Callable[[str, str | Path], tuple[pd.DataFrame, pd.DataFrame]],
    run_hmm2: Callable[..., dict[str, Any]],
    test_dir: Path,
    hmm_params: dict[str, Any],
    self_gr: dict[str, Any] | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for well in list_well_ids(test_dir):
        horizontal, typewell = load_well(well, test_dir)
        known = pd.to_numeric(horizontal["TVT_input"], errors="coerce").notna().to_numpy()
        if not known.any():
            raise ValueError(f"raw test well {well} has no finite TVT_input prefix")
        expected_eval = np.flatnonzero(~known).astype(np.int64)
        if len(expected_eval) == 0:
            continue
        kwargs = dict(hmm_params)
        if self_gr is not None:
            kwargs.update(
                {
                    "self_gr_config": dict(self_gr["surface"]),
                    "self_gr_alpha": float(self_gr["alpha"]),
                    "self_gr_clip": float(self_gr["clip"]),
                    "self_gr_mode": str(self_gr["mode"]),
                }
            )
        result = run_hmm2(horizontal, typewell, **kwargs)
        actual_eval = np.asarray(result["ev_index"], dtype=np.int64)
        if not np.array_equal(actual_eval, expected_eval):
            raise ValueError(f"HMM eval identity mismatch for well {well}")
        rows.append(
            pd.DataFrame(
                {
                    "id": [f"{well}_{int(row)}" for row in actual_eval],
                    "well": str(well),
                    "well_row_idx": actual_eval.astype(np.int32),
                    "candidate_tvt": np.asarray(result["mean_eval"], dtype=np.float32),
                    "sigma_tvt": np.asarray(result["std_eval"], dtype=np.float32),
                    "source_loglik": np.full(
                        len(actual_eval), np.float32(result["loglik"]), dtype=np.float32
                    ),
                    "loglik_per_row": np.full(
                        len(actual_eval),
                        np.float32(float(result["loglik"]) / len(actual_eval)),
                        dtype=np.float32,
                    ),
                }
            )
        )
        if self_gr is not None:
            rows[-1]["candidate_finite_source"] = np.isfinite(
                np.asarray(result["mean_eval"], dtype=np.float32)
            ).astype(np.float32)
            rows[-1]["selfgr_quality"] = np.asarray(
                result["self_gr_quality"], dtype=np.float32
            )
            rows[-1]["selfgr_peak_tvt"] = np.asarray(
                result["self_gr_peak_tvt"], dtype=np.float32
            )
            rows[-1]["score_margin"] = np.asarray(
                result["self_gr_peak_gap"], dtype=np.float32
            )
            rows[-1]["selfgr_typewell_agreement"] = np.asarray(
                result["self_gr_typewell_agreement"], dtype=np.float32
            )
            rows[-1]["selfgr_valid"] = np.asarray(
                result["self_gr_valid"], dtype=np.float32
            )
    if not rows:
        raise ValueError("HMM raw-test generation produced no rows")
    output = finalize_primitive_confidence(pd.concat(rows, ignore_index=True))
    if output.duplicated("id").any() or not np.isfinite(output["candidate_tvt"]).all():
        raise ValueError("HMM raw-test output violates duplicate/finite contract")
    return output


def generate_k16_primitive(
    module: Any,
    *,
    train_dir: Path,
    test_dir: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the pinned exp226 implementation once and retain its native GR delta."""
    params = module.params_from_config(config)
    max_train = get_nested(config, "inference.max_train_wells")
    max_test = get_nested(config, "inference.max_test_wells")
    train_wells = module.load_train_wells(
        train_dir,
        params,
        max_wells=int(max_train) if max_train is not None else None,
    )
    test_wells = module.load_test_wells(
        test_dir,
        params,
        max_wells=int(max_test) if max_test is not None else None,
    )
    if not train_wells or not test_wells:
        raise FileNotFoundError("exp226 K16 requires non-empty train and test wells")
    fields = module.build_fields(train_wells, params)
    kappa = module.fit_kappa(train_wells, fields, params)
    print("exp226 kappa:", np.round(kappa, 3))

    rows: list[pd.DataFrame] = []
    well_summaries: list[dict[str, Any]] = []
    for order, well in enumerate(test_wells, start=1):
        result = module.predict_well(well, fields, kappa, params)
        row_idx = np.arange(well.s + 1, well.s + well.n + 1, dtype=np.int32)
        if len(row_idx) != len(result.pred) or len(result.pred) != len(result.delta):
            raise ValueError(f"exp226 K16 row contract mismatch for well={well.wid}")
        rows.append(
            pd.DataFrame(
                {
                    "id": [f"{well.wid}_{int(row)}" for row in row_idx],
                    "well": str(well.wid),
                    "well_row_idx": row_idx,
                    "candidate_tvt": np.asarray(result.pred, dtype=np.float32),
                    "geometry_gr_delta": np.asarray(result.delta, dtype=np.float32),
                }
            )
        )
        summary = dict(result.summary)
        summary["order"] = order
        well_summaries.append(summary)
        print(
            f"exp226 {order}/{len(test_wells)} {well.wid}: "
            f"rows={well.n} delta_med={summary['delta_abs_median']:.3f}"
        )
    output = finalize_primitive_confidence(pd.concat(rows, ignore_index=True))
    if output.duplicated("id").any() or not np.isfinite(
        output[["candidate_tvt", "geometry_gr_delta"]].to_numpy()
    ).all():
        raise ValueError("exp226 K16 output violates duplicate/finite confidence contract")
    return output, {
        "experiment": str(get_nested(config, "experiment.name")),
        "status": "inference_with_native_confidence_completed",
        "train_wells": len(train_wells),
        "test_wells": len(test_wells),
        "rows": len(output),
        "kappa": [float(value) for value in np.asarray(kappa).ravel()],
        "well_summaries": well_summaries,
        "prediction_and_confidence_content_sha256": frame_content_sha256(output),
    }


def source_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def compare_current_reference(
    generated: pd.DataFrame,
    reference_path: Path,
    tolerance: float,
) -> dict[str, Any]:
    columns = {
        "exp226_k16": "exp226_v6_k16_geometry_gr_u_projection",
        "selfgr_hmm_a070": "hmm_selfgr_boost_only_a070_c100_mean_tvt",
        "likpf_mean": "likpf_mean",
        "exact_hmm": "hmm_exact_mean_tvt",
        "pf_ancc": "pf_ancc",
        "beam_mean": "beam_mean",
    }
    reference = pd.read_csv(
        reference_path,
        usecols=["id", *columns.values()],
        dtype={"id": str},
        low_memory=False,
    )
    generated_ids = set(generated["id"].astype(str))
    reference_ids = set(reference["id"].astype(str))
    if generated_ids != reference_ids:
        return {
            "status": "skipped_hidden_id_set_differs_from_current_reference",
            "generated_rows": len(generated),
            "reference_rows": len(reference),
        }
    reference = reference.rename(
        columns={
            reference_column: candidate_id
            for candidate_id, reference_column in columns.items()
        }
    )
    aligned = generated[["id", *columns]].merge(
        reference[["id", *columns]],
        on="id",
        how="left",
        validate="one_to_one",
        suffixes=("_generated", "_reference"),
    )
    max_abs: dict[str, float] = {}
    for candidate_id in columns:
        diff = np.abs(
            aligned[f"{candidate_id}_generated"].to_numpy(np.float64)
            - aligned[f"{candidate_id}_reference"].to_numpy(np.float64)
        )
        max_abs[candidate_id] = float(diff.max(initial=0.0))
    failed = {key: value for key, value in max_abs.items() if value > tolerance}
    if failed:
        raise ValueError(f"current-test source-port parity failed: {failed}")
    return {"status": "passed", "max_abs": max_abs, "tolerance": tolerance}


# %% [markdown]
# ## 3. Setup and submission contract

# %%
started = time.time()
paths = ExperimentPaths()
paths.require_kaggle_runtime()
paths.ensure_output_dirs()
config = load_config()
output_dir = paths.artifacts_dir
stage1 = get_nested(config, "stage1") or {}
contract = validate_contract()
rawtest_pairs = [pair for pair in PAIR_SHORTLIST if pair.tier == "raw-test"]

if not bool(stage1.get("enabled")):
    raise ValueError("Stage 1 must be enabled for submission inference")
if not bool(stage1.get("create_submission")):
    raise ValueError("Stage 1 create_submission must be true")
if stage1.get("selected_submission_candidate") != "exp226_w500_50_50":
    raise ValueError("exp263 submission is fixed to exp226_w500_50_50")
if len(RAWTEST_CORE_CANDIDATE_IDS) != 6 or len(rawtest_pairs) != 5:
    raise ValueError("Stage 1 deployability tier count mismatch")
configured_confidence = stage1["confidence_output"]["required_fields_by_primitive"]
expected_confidence = {
    candidate_id: ["confidence_valid", *fields]
    for candidate_id, fields in STAGE1_NATIVE_CONFIDENCE_FIELDS.items()
}
if configured_confidence != expected_confidence:
    raise ValueError("Stage 1 config/native confidence contract mismatch")

sample = pd.read_csv(paths.sample_submission_path, dtype={"id": str})
display(
    {
        "experiment": EXPERIMENT_NAME,
        "route": get_nested(config, "experiment.route"),
        "mode": stage1.get("mode"),
        "contract": contract,
        "selected_submission": stage1.get("selected_submission_candidate"),
        "selected_oof_rmse": stage1.get("selected_submission_oof_rmse"),
        "weights": NAMED_COMBINATIONS["exp226_w500_50_50"]["weights"],
        "sample_rows": len(sample),
        "gpu": get_nested(config, "runtime.kaggle.enable_gpu"),
        "internet": get_nested(config, "runtime.kaggle.internet"),
        "training_variants": 0,
        "boosters": 0,
    }
)

# %% [markdown]
# ## 4. Trusted upstream source resolution

# %%
generation = stage1["raw_test_generation"]
source_work = output_dir / "trusted_upstream_sources"
source_specs = {
    "exp263_public_replay_source": generation["pf_replay"],
    "exp263_exact_hmm_source": generation["exact_hmm"],
    "exp263_selfgr_hmm_source": generation["selfgr_hmm_a070"],
    "exp263_k16_source": generation["exp226_k16"],
}
resolved_sources: dict[str, Path] = {}
copied_sources: dict[str, Path] = {}
for module_name, spec in source_specs.items():
    source = resolve_unique_source(spec["source_filename"], spec["source_path_token"])
    resolved_sources[module_name] = source
    copied_sources[module_name] = copy_trusted_source(source, source_work, module_name)

sys.path.insert(0, str(source_work))
import exp263_k16_source as k16_module  # noqa: E402
from exp263_exact_hmm_source import (  # noqa: E402
    list_well_ids as exact_list_well_ids,
)
from exp263_exact_hmm_source import load_well as exact_load_well  # noqa: E402
from exp263_exact_hmm_source import run_hmm2 as exact_run_hmm2  # noqa: E402
from exp263_public_replay_source import (  # noqa: E402
    build_replay_test_frame,
    configure_public_runtime,
)
from exp263_selfgr_hmm_source import (  # noqa: E402
    list_well_ids as selfgr_list_well_ids,
)
from exp263_selfgr_hmm_source import load_well as selfgr_load_well  # noqa: E402
from exp263_selfgr_hmm_source import run_hmm2 as selfgr_run_hmm2  # noqa: E402

stage0_manifest = resolve_unique_source(
    "cache_manifest.json", "exp263-last-anchor-pair-cache-train"
)
expected_manifest_sha = stage1["stage0_manifest"]["expected_manifest_sha256"]
if sha256_file(stage0_manifest) != expected_manifest_sha:
    raise ValueError("Stage 0 manifest SHA differs from the fixed exp263 contract")

source_audit = {name: source_record(path) for name, path in resolved_sources.items()}
source_audit["stage0_manifest"] = source_record(stage0_manifest)
display(source_audit)

# %% [markdown]
# ## 5. Raw-test six-primitive regeneration

# %%
pf_config = generation["pf_replay"]
configure_public_runtime(
    data_dir=paths.raw_data_dir,
    output_dir=output_dir / "pf_replay",
    n_jobs=int(pf_config["n_jobs"]),
    pf_seeds=int(pf_config["pf_seeds"]),
    pf_particles=int(pf_config["pf_particles"]),
    fast=bool(pf_config["fast"]),
    use_gpu=str(pf_config["use_gpu"]),
)
pf_frame, pf_meta = build_replay_test_frame()
required_pf = {
    "id",
    "well",
    "last_known_tvt",
    "likpf_mean_d",
    "pf_ancc",
    "pf_ancc_std",
    "beam_mean_d",
    "beam_std_d",
}
if missing_pf := required_pf - set(pf_frame.columns):
    raise ValueError(f"exp073 raw-test replay columns missing: {sorted(missing_pf)}")

k16_source_config = resolved_sources["exp263_k16_source"].parent / generation[
    "exp226_k16"
]["source_config_filename"]
if not k16_source_config.exists():
    raise FileNotFoundError(f"exp226 source config missing: {k16_source_config}")
k16_config = yaml.safe_load(k16_source_config.read_text())
k16_frame, k16_summary = generate_k16_primitive(
    k16_module,
    train_dir=paths.train_data_dir,
    test_dir=paths.test_data_dir,
    config=k16_config,
)

exact_config = generation["exact_hmm"]
exact_frame = generate_hmm_primitive(
    list_well_ids=exact_list_well_ids,
    load_well=exact_load_well,
    run_hmm2=exact_run_hmm2,
    test_dir=paths.test_data_dir,
    hmm_params=dict(exact_config["params"]),
)
selfgr_config = generation["selfgr_hmm_a070"]
selfgr_frame = generate_hmm_primitive(
    list_well_ids=selfgr_list_well_ids,
    load_well=selfgr_load_well,
    run_hmm2=selfgr_run_hmm2,
    test_dir=paths.test_data_dir,
    hmm_params=dict(exact_config["params"]),
    self_gr=dict(selfgr_config),
)

primitive_frames = {
    "exp226_k16": k16_frame,
    "selfgr_hmm_a070": selfgr_frame,
    "likpf_mean": standard_primitive(
        pf_frame,
        pf_frame["last_known_tvt"].to_numpy(np.float32)
        + pf_frame["likpf_mean_d"].to_numpy(np.float32),
    ),
    "exact_hmm": exact_frame,
    "pf_ancc": standard_primitive(
        pf_frame,
        pf_frame["pf_ancc"],
        confidence={"sigma_tvt": pf_frame["pf_ancc_std"]},
    ),
    "beam_mean": standard_primitive(
        pf_frame,
        pf_frame["last_known_tvt"].to_numpy(np.float32)
        + pf_frame["beam_mean_d"].to_numpy(np.float32),
        confidence={"beam_family_std": pf_frame["beam_std_d"]},
    ),
}
display(
    pd.DataFrame(
        [
            {
                "candidate_id": candidate_id,
                "rows": len(frame),
                "wells": frame["well"].nunique(),
                "finite": bool(np.isfinite(frame["candidate_tvt"]).all()),
                "content_sha256": frame_content_sha256(frame),
            }
            for candidate_id, frame in primitive_frames.items()
        ]
    )
)

# %% [markdown]
# ## 6. Five pair and fixed named-formula parity

# %%
formula_frame, max_abs_formula = assemble_stage1_current_test_parity(primitive_frames)
formula_frame = attach_stage1_current_test_confidence(formula_frame, primitive_frames)
formula_path = output_dir / "current_test_formula_parity.parquet"
formula_frame.to_parquet(formula_path, index=False, compression="zstd")

expected_formula_columns = {
    *RAWTEST_CORE_CANDIDATE_IDS,
    *(pair.pair_id for pair in rawtest_pairs),
    "exp226_w500_50_50",
}
missing_formula_columns = expected_formula_columns - set(formula_frame.columns)
if missing_formula_columns:
    raise ValueError(f"Stage 1 formula columns missing: {sorted(missing_formula_columns)}")
display(formula_frame.head(10))
display(formula_frame[list(expected_formula_columns)].describe().T)
confidence_coverage = {
    candidate_id: {
        "fields": ["confidence_valid", *STAGE1_NATIVE_CONFIDENCE_FIELDS[candidate_id]],
        "valid_rate": float(frame["confidence_valid"].astype(bool).mean()),
    }
    for candidate_id, frame in primitive_frames.items()
}
if sum(column.startswith("confidence__") for column in formula_frame.columns) != 21:
    raise ValueError("Stage 1 must export exactly 21 namespaced confidence columns")
display(confidence_coverage)

# %% [markdown]
# ## 7. Current-test reference parity and submission generation

# %%
reference_config = stage1["current_test_reference"]
reference_path = resolve_unique_source(
    "exp237_hmm_exp226_candidate_selector_on_exp183_rawtest_candidate_features.csv.gz",
    "exp237-hmm-exp226-candidate-selector-exp183-infer",
)
reference_parity = compare_current_reference(
    formula_frame,
    reference_path,
    tolerance=float(reference_config["max_abs_tolerance"]),
)

submission = build_submission_from_stage1_parity(
    sample,
    formula_frame,
    candidate_id=str(stage1["selected_submission_candidate"]),
)
submission.to_csv(paths.submission_path, index=False)
if len(submission) != len(sample) or not submission["id"].equals(sample["id"]):
    raise ValueError("submission row/order contract failed")
if submission["tvt"].isna().any() or not np.isfinite(submission["tvt"]).all():
    raise ValueError("submission finite contract failed")

display(reference_parity)
display(submission.head(20))
display(submission["tvt"].describe())

# %% [markdown]
# ## 8. SHA, metrics, and generated artifacts

# %%
primitive_sha = {
    candidate_id: frame_content_sha256(frame)
    for candidate_id, frame in primitive_frames.items()
}
metrics = {
    "experiment": EXPERIMENT_NAME,
    "route": get_nested(config, "experiment.route"),
    "stage": "stage1_hidden_safe_submission",
    "status": "stage1_hidden_safe_submission_completed",
    "selected_candidate": stage1["selected_submission_candidate"],
    "selected_oof_rmse": float(stage1["selected_submission_oof_rmse"]),
    "weights": NAMED_COMBINATIONS["exp226_w500_50_50"]["weights"],
    "rows": len(submission),
    "wells": int(formula_frame["well"].nunique()),
    "rawtest_primitive_count": 6,
    "rawtest_pair_count": 5,
    "namespaced_confidence_column_count": sum(
        column.startswith("confidence__") for column in formula_frame.columns
    ),
    "confidence_coverage": confidence_coverage,
    "max_abs_formula_parity": max_abs_formula,
    "current_test_reference_parity": reference_parity,
    "pf_generation": pf_meta,
    "exp226_generation": k16_summary,
    "source_audit": source_audit,
    "stage0_manifest_sha256": sha256_file(stage0_manifest),
    "primitive_content_sha256": primitive_sha,
    "formula_parity_file_sha256": sha256_file(formula_path),
    "prediction_content_sha256": frame_content_sha256(
        submission.rename(columns={"tvt": "prediction"})
    ),
    "submission_sha256": sha256_file(paths.submission_path),
    "prediction_stats": {
        "min": float(submission["tvt"].min()),
        "max": float(submission["tvt"].max()),
        "mean": float(submission["tvt"].mean()),
        "std": float(submission["tvt"].std()),
    },
    "runtime_seconds": round(time.time() - started, 3),
    "model_sha": "not_applicable_no_training",
    "booster_count": 0,
}
metrics_text = json.dumps(metrics, indent=2, ensure_ascii=False, default=str) + "\n"
(output_dir / "stage1_metrics.json").write_text(metrics_text)
paths.metrics_path.write_text(metrics_text)
(output_dir / "stage1_summary.json").write_text(metrics_text)
display(metrics)

print("Generated artifacts:")
for generated_path in [formula_path, paths.submission_path, output_dir / "stage1_metrics.json"]:
    print(f"- {generated_path} ({generated_path.stat().st_size} bytes)")
