from __future__ import annotations

import ast
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "exp490_geometry_centered_mean_reverting_offset_hmm"
EXPERIMENT_DIR = ROOT / "experiments" / EXPERIMENT
TRAIN_SOURCE = EXPERIMENT_DIR / f"{EXPERIMENT}_train_aggregate.py"
OUTPUT_SOURCE = EXPERIMENT_DIR / f"{EXPERIMENT}_compact_selfcontained_inference.py"

EXTRACT_NAMES = (
    "get_nested",
    "to_jsonable",
    "stable_json_bytes",
    "sha256_file",
    "logical_frame_sha256",
    "array_bundle_sha256",
    "write_json",
    "write_csv",
    "write_deterministic_gzip_csv",
    "peak_rss_gib",
    "runtime_versions",
    "huber_log_likelihood",
    "robust_initial_rate",
    "prefix_stats",
    "k16_segment_half_life",
    "zero_state_geometry_identity",
    "prepare_hmm_inputs",
    "_hmm2_fb_geometry_mean_reversion",
    "run_geometry_mean_reverting_hmm",
    "FrozenWell",
    "prediction_frame",
    "segment_contract_frame",
)


HEADER = '''# %% [markdown]
# # exp490 geometry-centered mean-reverting offset HMM inference
#
# This notebook is an explicit Public-LB audit after the Stage 1 well-tail gate
# failed. It does not promote or alter exp490. It regenerates the SHA-pinned
# exp226 geometry-only field on the three current-test wells, applies exactly
# the frozen exp490 mean-reverting Huber HMM, and writes a submission candidate.
# Competition submission is outside this notebook.

# %% [markdown]
# ## Contents
# 1. Imports and frozen execution contract
# 2. Notebook-safe paths, SHA, and scientific-contract helpers
# 3. Trusted exp226 geometry-only current-test regeneration
# 4. Raw-test input and K16 half-life helpers
# 5. Frozen geometry-centered exact forward-backward decoder
# 6. Per-well prediction and generated-artifact helpers
# 7. Setup and input preflight
# 8. Generate the three-well current-test candidate
# 9. Submission, technical gates, SHA, and summary

# %% [markdown]
# ## 1. Imports and frozen execution contract

# %%
from __future__ import annotations

import gc
import gzip
import hashlib
import io
import json
import math
import os
import platform
import resource
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

try:
    from numba import njit, prange, set_num_threads

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover
    NUMBA_AVAILABLE = False

    def njit(*args: Any, **kwargs: Any):  # type: ignore[misc]
        del args, kwargs

        def decorator(function):
            return function

        return decorator

    def prange(*args: int):  # type: ignore[misc]
        return range(*args)

    def set_num_threads(_: int) -> None:
        return None


EXPERIMENT_NAME = "exp490_geometry_centered_mean_reverting_offset_hmm"
PACKAGE_DIR = Path.cwd()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
EXPECTED_OOF_SCIENTIFIC_CONTRACT_SHA256 = (
    "6398bbac380d3eca3a6255681b22c44c26de268ce6d4fad9dd242c066f2b9a35"
)
FORBIDDEN_HORIZONTAL_COLUMNS = {"TVT", "tvt_true", "error", "abs_error", "fold"}
'''


HELPERS = '''
# %% [markdown]
# ## 2. Notebook-safe paths, SHA, and scientific-contract helpers

# %%
def find_project_root(start: Path = PACKAGE_DIR) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists() and (candidate / "experiments").is_dir():
            return candidate
    return start


def config_path() -> Path:
    candidates = (
        PACKAGE_DIR / "config.yaml",
        find_project_root() / "experiments" / EXPERIMENT_NAME / "config.yaml",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("exp490 config.yaml not found")


def load_config() -> dict[str, Any]:
    payload = yaml.safe_load(config_path().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config.yaml must contain a mapping")
    return payload


def artifacts_dir() -> Path:
    path = KAGGLE_WORKING_ROOT / "artifacts" if KAGGLE_WORKING_ROOT.is_dir() else PACKAGE_DIR / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def metrics_path() -> Path:
    return KAGGLE_WORKING_ROOT / "metrics.json" if KAGGLE_WORKING_ROOT.is_dir() else PACKAGE_DIR / "metrics.json"


def submission_path() -> Path:
    return KAGGLE_WORKING_ROOT / "submission.csv" if KAGGLE_WORKING_ROOT.is_dir() else PACKAGE_DIR / "submission.csv"


def require_kaggle_runtime() -> None:
    if not KAGGLE_INPUT_ROOT.is_dir() or not KAGGLE_WORKING_ROOT.is_dir():
        raise RuntimeError("exp490 inference must run on Kaggle")
    if not os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        raise RuntimeError("KAGGLE_KERNEL_RUN_TYPE is required")


def resolve_data_root(config: Mapping[str, Any]) -> Path:
    local = find_project_root() / str(get_nested(config, "data.raw_dir"))
    if (local / "sample_submission.csv").is_file() and (local / "test").is_dir():
        return local
    fixed = (
        KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction",
        KAGGLE_INPUT_ROOT / "competitions" / "rogii-wellbore-geology-prediction",
    )
    for candidate in fixed:
        if (candidate / "sample_submission.csv").is_file() and (candidate / "test").is_dir():
            return candidate
    matches = sorted(
        path.parent
        for path in KAGGLE_INPUT_ROOT.rglob("sample_submission.csv")
        if (path.parent / "train").is_dir() and (path.parent / "test").is_dir()
    )
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"competition data root resolution failed: {matches}")


def parse_sample_identity(sample: pd.DataFrame) -> pd.DataFrame:
    if list(sample.columns) != ["id", "tvt"]:
        raise ValueError(f"sample columns changed: {list(sample.columns)}")
    if sample["id"].duplicated().any():
        raise ValueError("sample ids are not unique")
    parts = sample["id"].astype(str).str.rsplit("_", n=1, expand=True)
    if parts.shape[1] != 2:
        raise ValueError("sample id must be '<well>_<row_idx>'")
    identity = pd.DataFrame(
        {
            "id": sample["id"].astype(str),
            "well": parts[0].astype(str),
            "row_idx": pd.to_numeric(parts[1], errors="raise").astype(np.int64),
        }
    )
    if identity.duplicated(["well", "row_idx"]).any():
        raise ValueError("sample well/row identity is not unique")
    return identity


def validate_inference_execution(config: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "scientific_variants": 1,
        "exp226_full_train_fits": 1,
        "exp226_test_geometry_well_runs": 3,
        "candidate_hmm_well_runs": 3,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "pf_runs": 0,
        "beam_runs": 0,
        "gpu_runs": 0,
    }
    observed = dict(get_nested(config, "execution_contract.current_test_inference_approved"))
    if observed != expected:
        raise ValueError(f"inference execution contract changed: {observed}")
    checks = {
        "run_inference": bool(get_nested(config, "execution.run_inference")),
        "create_submission": bool(get_nested(config, "execution.create_submission")),
        "inference_run_approved": bool(get_nested(config, "execution.inference_run_approved")),
        "inference_enabled": bool(get_nested(config, "inference.enabled")),
        "preserve_fail_close": bool(get_nested(config, "inference.preserve_stage_1_fail_close")),
        "stage_1_failed": not bool(get_nested(config, "implementation.stage_1_all_pass")),
        "competition_submission_disabled": not bool(
            get_nested(config, "execution.competition_submission_approved")
        ),
        "gpu_disabled": not bool(get_nested(config, "runtime.kaggle.enable_gpu")),
        "internet_disabled": not bool(get_nested(config, "runtime.kaggle.enable_internet")),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"inference authorization contract failed: {failed}")
    return {**expected, "execution_basis": "explicit_user_lb_audit_after_stage1_tail_fail"}


def build_frozen_oof_scientific_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "experiment": get_nested(config, "experiment.name") == EXPERIMENT_NAME,
        "route": get_nested(config, "experiment.route") == "pf_beam",
        "parent": get_nested(config, "lineage.parent") == "exp357_exp226_huber_emission_independent_audit",
        "variant": get_nested(config, "model.mean_reversion.active_variants") == ["k16_segment_span_half_life"],
        "k16": int(get_nested(config, "model.k16_segment.count")) == 16,
        "half_life": float(get_nested(config, "model.mean_reversion.half_life_segments")) == 1.0,
        "delta_min": float(get_nested(config, "model.hmm.delta_min_ft")) == -80.0,
        "delta_max": float(get_nested(config, "model.hmm.delta_max_ft")) == 80.0,
        "step": float(get_nested(config, "model.hmm.step_ft")) == 0.35,
        "rates": int(get_nested(config, "model.hmm.n_rates")) == 41,
        "rate_span": float(get_nested(config, "model.hmm.rate_span")) == 0.10,
        "sig_r": float(get_nested(config, "model.hmm.sig_r")) == 0.002,
        "sig_p": float(get_nested(config, "model.hmm.sig_p")) == 0.02,
        "momentum": float(get_nested(config, "model.hmm.mom")) == 0.998,
        "huber": float(get_nested(config, "model.emission.huber_delta")) == 1.345,
        "hard_reset_disabled": not bool(get_nested(config, "model.mean_reversion.hard_reset")),
        "gr_gate_disabled": not bool(get_nested(config, "model.mean_reversion.gr_confidence_gate")),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"frozen scientific model changed: {failed}")
    contract = {
        "experiment": EXPERIMENT_NAME,
        "parent": get_nested(config, "lineage.parent"),
        "route": "pf_beam",
        "stage": "stage_1_full_oof_four_target_free_shards_then_truth_late_merge",
        "candidate_variants": 1,
        "coordinate": {
            "prediction": "exp226_tvt_geop_t + residual_offset_t",
            "rate_center": "0.998 * rho_t * q_previous",
            "offset_center": "rho_t * residual_offset_previous + q_t * positive_dMD_t",
        },
        "rho": {
            "formula": "2 ** (-dMD_t / destination_K16_segment_MD_span)",
            "boundary_owner": "destination_row_segment",
            "half_life_segments": 1.0,
        },
        "hmm": dict(get_nested(config, "model.hmm")),
        "emission": dict(get_nested(config, "model.emission")),
        "forbidden_rescue": list(get_nested(config, "model.forbidden_rescue")),
        "truth_attachment": get_nested(config, "validation.truth_attachment"),
        "control": "SHA-pinned saved exp357 prediction; zero HMM reruns",
    }
    observed = hashlib.sha256(stable_json_bytes(contract)).hexdigest()
    expected = str(get_nested(config, "inference.expected_scientific_contract_sha256"))
    if observed != EXPECTED_OOF_SCIENTIFIC_CONTRACT_SHA256 or observed != expected:
        raise ValueError(f"OOF scientific contract SHA changed: {observed}")
    return contract


def resolve_exp226_source(config: Mapping[str, Any]) -> tuple[Path, Path, dict[str, Any]]:
    spec = dict(get_nested(config, "data.exp226_inference_source"))
    source_name = str(spec["source_filename"])
    config_name = str(spec["config_filename"])
    local_source = (
        find_project_root()
        / "experiments"
        / "exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction"
        / source_name
    )
    local_output = Path(
        "/tmp/kaggle-output/exp226_connortynan_k16_spline_kernel_knn_adaptive_kappa_reproduction/inference_v1"
    )
    candidates = []
    if local_output.joinpath(source_name).is_file():
        candidates.append(local_output / source_name)
    if KAGGLE_INPUT_ROOT.is_dir():
        candidates.extend(KAGGLE_INPUT_ROOT.rglob(source_name))
    if local_source.is_file():
        candidates.append(local_source)
    valid = []
    for source in candidates:
        sibling = source.parent / config_name
        if (
            sibling.is_file()
            and sha256_file(source) == str(spec["source_sha256"])
            and sha256_file(sibling) == str(spec["config_sha256"])
        ):
            valid.append((source, sibling))
    unique = {(str(source.resolve()), str(cfg.resolve())): (source, cfg) for source, cfg in valid}
    if not unique:
        raise FileNotFoundError("SHA-pinned exp226 inference source/config pair not found")
    source, source_config = sorted(unique.values(), key=lambda pair: str(pair[0]))[0]
    return source, source_config, {
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "config_path": str(source_config),
        "config_sha256": sha256_file(source_config),
    }
'''


EXP226_AND_RAW = '''
# %% [markdown]
# ## 3. Trusted exp226 geometry-only current-test regeneration

# %%
def generate_exp226_test_geometry(
    config: Mapping[str, Any],
    data_root: Path,
    expected_wells: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    source, source_config_path, source_audit = resolve_exp226_source(config)
    trusted_module = KAGGLE_WORKING_ROOT / "exp490_trusted_exp226_k16.py"
    shutil.copyfile(source, trusted_module)
    if str(KAGGLE_WORKING_ROOT) not in sys.path:
        sys.path.insert(0, str(KAGGLE_WORKING_ROOT))
    import exp490_trusted_exp226_k16 as k16_module

    exp226_config = yaml.safe_load(source_config_path.read_text(encoding="utf-8"))
    params = k16_module.params_from_config(exp226_config)
    train_wells = k16_module.load_train_wells(data_root / "train", params)
    test_wells = k16_module.load_test_wells(data_root / "test", params)
    if len(train_wells) != 773:
        raise ValueError(f"exp226 full-train well count changed: {len(train_wells)}")
    observed_wells = sorted(well.wid for well in test_wells)
    if observed_wells != sorted(str(well) for well in expected_wells):
        raise ValueError(f"exp226 test wells changed: {observed_wells}")
    fields = k16_module.build_fields(train_wells, params)
    kappa = k16_module.fit_kappa(train_wells, fields, params)
    pieces = []
    well_rows = []
    for order, well in enumerate(test_wells, start=1):
        result = k16_module.predict_well(well, fields, kappa, params)
        if len(result.geop) != len(well.suffix_row_idx):
            raise ValueError(f"{well.wid}: exp226 geop row mismatch")
        frame = pd.DataFrame(
            {
                "id": [f"{well.wid}_{int(row)}" for row in well.suffix_row_idx],
                "well": str(well.wid),
                "row_idx": np.asarray(well.suffix_row_idx, dtype=np.int64),
                "suffix_offset": np.arange(len(result.geop), dtype=np.int64),
                "tvt_geop": np.asarray(result.geop, dtype=np.float64),
            }
        )
        if not np.isfinite(frame["tvt_geop"]).all():
            raise ValueError(f"{well.wid}: exp226 geop contains non-finite values")
        pieces.append(frame)
        well_rows.append(
            {
                "order": order,
                "well": str(well.wid),
                "prefix_rows": int(well.s + 1),
                "unknown_rows": int(len(result.geop)),
                "geop_min": float(np.min(result.geop)),
                "geop_max": float(np.max(result.geop)),
            }
        )
    geometry = pd.concat(pieces, ignore_index=True)
    kappa_frame = pd.DataFrame(k16_module.kappa_terms(kappa, params, fold="full_train"))
    return geometry, kappa_frame, {
        **source_audit,
        "train_wells": len(train_wells),
        "test_wells": len(test_wells),
        "geometry_rows": len(geometry),
        "geometry_logical_sha256": logical_frame_sha256(geometry),
        "geometry_attribute": "PredictionResult.geop",
        "forbidden_exp226_outputs_used": [],
        "wells": well_rows,
    }


# %% [markdown]
# ## 4. Raw-test input and K16 half-life helpers

# %%
def load_target_free_test_well(
    well: str,
    test_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    horizontal_path = test_dir / f"{well}__horizontal_well.csv"
    typewell_path = test_dir / f"{well}__typewell.csv"
    horizontal = pd.read_csv(horizontal_path, usecols=lambda column: str(column) != "TVT")
    forbidden = FORBIDDEN_HORIZONTAL_COLUMNS.intersection(horizontal.columns)
    if forbidden:
        raise ValueError(f"{well}: forbidden test columns {sorted(forbidden)}")
    typewell = pd.read_csv(typewell_path).sort_values("TVT", kind="mergesort").reset_index(drop=True)
    if not {"TVT", "GR"}.issubset(typewell.columns):
        raise ValueError(f"{well}: typewell requires TVT and GR")
    return horizontal, typewell, {
        "horizontal_sha256": sha256_file(horizontal_path),
        "typewell_sha256": sha256_file(typewell_path),
    }
'''


FREEZE_AND_RUN = '''
# %% [markdown]
# ## 6. Per-well prediction and generated-artifact helpers

# %%
def freeze_test_well(
    *,
    well: str,
    test_dir: Path,
    exp226_well: pd.DataFrame,
    config: Mapping[str, Any],
) -> FrozenWell:
    horizontal, typewell, raw_sha = load_target_free_test_well(well, test_dir)
    evaluation_index = horizontal.index[horizontal["TVT_input"].isna()].to_numpy(np.int64)
    source = exp226_well.sort_values("row_idx", kind="mergesort")
    source_rows = source["row_idx"].to_numpy(np.int64)
    suffix_offset = source["suffix_offset"].to_numpy(np.int64)
    if not np.array_equal(source_rows, evaluation_index):
        raise ValueError(f"{well}: exp226/raw row identity mismatch")
    if not np.array_equal(suffix_offset, np.arange(len(source), dtype=np.int64)):
        raise ValueError(f"{well}: suffix offsets are not contiguous")
    geop = source["tvt_geop"].to_numpy(np.float64)
    started = time.perf_counter()
    result = run_geometry_mean_reverting_hmm(horizontal, typewell, geop, config)
    elapsed = float(time.perf_counter() - started)
    arrays = (geop, result["mean"], result["delta_mean"], result["std"], result["dmd"], result["rho"])
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError(f"{well}: non-finite inference output")
    prediction_sha = array_bundle_sha256(
        row_idx=source_rows,
        suffix_offset=suffix_offset,
        tvt_geop=geop,
        prediction=np.asarray(result["mean"], dtype=np.float64),
        delta_mean=np.asarray(result["delta_mean"], dtype=np.float64),
        posterior_std=np.asarray(result["std"], dtype=np.float64),
        dmd=np.asarray(result["dmd"], dtype=np.float64),
        segment_id=np.asarray(result["segment_id"], dtype=np.int16),
        rho=np.asarray(result["rho"], dtype=np.float64),
    )
    return FrozenWell(
        well=str(well),
        row_idx=source_rows,
        suffix_offset=suffix_offset,
        tvt_geop=geop,
        prediction=np.asarray(result["mean"], dtype=np.float64),
        delta_mean=np.asarray(result["delta_mean"], dtype=np.float64),
        posterior_std=np.asarray(result["std"], dtype=np.float64),
        dmd=np.asarray(result["dmd"], dtype=np.float64),
        segment_id=np.asarray(result["segment_id"], dtype=np.int16),
        segment_span=np.asarray(result["segment_span"], dtype=np.float64),
        segment_rows=np.asarray(result["segment_rows"], dtype=np.int64),
        segment_cumulative_rho=np.asarray(result["segment_cumulative_rho"], dtype=np.float64),
        rho=np.asarray(result["rho"], dtype=np.float64),
        prefix_rows=int(result["prefix_rows"]),
        prefix_sigma=float(result["prefix_sigma"]),
        maximum_posterior_normalization_error=float(result["maximum_posterior_normalization_error"]),
        zero_state_identity_pass=bool(result["zero_state_identity"]["pass"]),
        emission_finite_coverage=float(result["emission_finite_coverage"]),
        elapsed_seconds=elapsed,
        prediction_sha256=prediction_sha,
        raw_input_sha256=raw_sha,
    )


def build_submission(sample: pd.DataFrame, prediction: pd.DataFrame) -> pd.DataFrame:
    if prediction["id"].duplicated().any():
        raise ValueError("candidate ids are not unique")
    mapping = prediction.set_index("id")["geometry_mean_reverting_hmm"]
    if set(mapping.index) != set(sample["id"].astype(str)):
        raise ValueError("candidate and sample id sets differ")
    submission = sample[["id"]].copy()
    submission["tvt"] = submission["id"].astype(str).map(mapping)
    if submission["tvt"].isna().any() or not np.isfinite(submission["tvt"]).all():
        raise ValueError("submission contains missing/non-finite values")
    if not submission["id"].equals(sample["id"]):
        raise ValueError("submission does not preserve sample order")
    return submission


def run_inference(config: Mapping[str, Any]) -> dict[str, Any]:
    require_kaggle_runtime()
    started = time.perf_counter()
    execution_contract = validate_inference_execution(config)
    scientific_contract = build_frozen_oof_scientific_contract(config)
    scientific_sha = hashlib.sha256(stable_json_bytes(scientific_contract)).hexdigest()
    set_num_threads(int(get_nested(config, "runtime.numba_num_threads")))
    data_root = resolve_data_root(config)
    sample_path = data_root / "sample_submission.csv"
    if sha256_file(sample_path) != str(get_nested(config, "data.sample_submission_expected_sha256")):
        raise ValueError("sample_submission SHA changed")
    sample = pd.read_csv(sample_path, dtype={"id": str})
    identity = parse_sample_identity(sample)
    expected_rows = int(get_nested(config, "data.expected_submission_rows"))
    expected_wells = int(get_nested(config, "data.expected_test_wells"))
    wells = sorted(identity["well"].unique().tolist())
    if len(sample) != expected_rows or len(wells) != expected_wells:
        raise ValueError("sample row/well count changed")

    geometry, kappa_frame, exp226_audit = generate_exp226_test_geometry(
        config, data_root, wells
    )
    if set(geometry["id"]) != set(identity["id"]):
        raise ValueError("exp226 geometry identity differs from sample")
    frozen = []
    for order, well in enumerate(wells, start=1):
        item = freeze_test_well(
            well=well,
            test_dir=data_root / "test",
            exp226_well=geometry.loc[geometry["well"] == well],
            config=config,
        )
        frozen.append(item)
        print(
            json.dumps(
                {
                    "event": "exp490_inference_well_complete",
                    "order": order,
                    "wells": len(wells),
                    "well": well,
                    "rows": len(item.row_idx),
                    "elapsed_seconds": item.elapsed_seconds,
                    "peak_rss_gib": peak_rss_gib(),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    prediction = prediction_frame(frozen)
    segment = segment_contract_frame(frozen)
    submission = build_submission(sample, prediction)
    submission.to_csv(submission_path(), index=False)
    if len(prediction) != expected_rows or prediction["well"].nunique() != expected_wells:
        raise ValueError("prediction row/well coverage changed")
    max_norm = float(max(item.maximum_posterior_normalization_error for item in frozen))
    max_half_life_error = float(segment["rho_product_abs_error_vs_half"].max())
    technical = {
        "exp226_source_sha": exp226_audit["source_sha256"] == str(get_nested(config, "data.exp226_inference_source.source_sha256")),
        "exp226_config_sha": exp226_audit["config_sha256"] == str(get_nested(config, "data.exp226_inference_source.config_sha256")),
        "scientific_contract_parity": scientific_sha == EXPECTED_OOF_SCIENTIFIC_CONTRACT_SHA256,
        "sample_row_and_well_coverage": len(prediction) == expected_rows and prediction["well"].nunique() == expected_wells,
        "sample_id_set": set(prediction["id"]) == set(sample["id"].astype(str)),
        "sample_order": submission["id"].equals(sample["id"]),
        "prediction_finite": bool(np.isfinite(prediction["geometry_mean_reverting_hmm"]).all()),
        "submission_finite": bool(np.isfinite(submission["tvt"]).all()),
        "posterior_normalization": max_norm <= 1.0e-6,
        "segment_half_life": max_half_life_error <= 1.0e-10,
        "positive_dmd": bool(segment["positive_dmd"].all()),
        "zero_state_geometry_identity": all(item.zero_state_identity_pass for item in frozen),
        "competition_submission_disabled": not bool(get_nested(config, "execution.competition_submission_approved")),
    }
    if not all(technical.values()):
        raise RuntimeError(f"exp490 inference technical gate failed: {technical}")

    out_dir = artifacts_dir()
    prediction_record = write_deterministic_gzip_csv(
        out_dir / f"{EXPERIMENT_NAME}_current_test_predictions.csv.gz",
        prediction,
    )
    segment_record = write_csv(
        out_dir / f"{EXPERIMENT_NAME}_current_test_segment_contract.csv",
        segment,
    )
    kappa_record = write_csv(
        out_dir / f"{EXPERIMENT_NAME}_current_test_exp226_kappa.csv",
        kappa_frame,
    )
    input_manifest = {
        "competition_data_root": str(data_root),
        "sample_submission_path": str(sample_path),
        "sample_submission_sha256": sha256_file(sample_path),
        "sample_rows": len(sample),
        "test_wells": wells,
        "exp226": exp226_audit,
        "raw_input_sha256_by_well": {item.well: item.raw_input_sha256 for item in frozen},
    }
    input_record = write_json(
        out_dir / f"{EXPERIMENT_NAME}_inference_input_manifest.json",
        input_manifest,
    )
    decoder_manifest = {
        "experiment": EXPERIMENT_NAME,
        "stage": "current_test_inference_lb_audit",
        "execution_basis": "explicit_user_lb_audit_after_stage1_tail_fail",
        "scientific_contract": scientific_contract,
        "scientific_contract_sha256": scientific_sha,
        "prediction_sha256_by_well": {item.well: item.prediction_sha256 for item in frozen},
        "state_order": "same_as_stage_1_full_oof",
        "competition_submission": False,
    }
    decoder_record = write_json(
        out_dir / f"{EXPERIMENT_NAME}_inference_decoder_manifest.json",
        decoder_manifest,
    )
    elapsed = float(time.perf_counter() - started)
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": "inference_complete_pending_submit_check_and_submission_approval",
        "route": "pf_beam",
        "execution_basis": "explicit_user_lb_audit_after_stage1_tail_fail",
        "stage_1_fail_close_preserved": True,
        "rows": len(submission),
        "wells": len(wells),
        "execution_contract": execution_contract,
        "scientific_contract_sha256": scientific_sha,
        "technical": technical,
        "maximum_posterior_normalization_error": max_norm,
        "maximum_segment_half_life_abs_error": max_half_life_error,
        "runtime": {
            "total_elapsed_seconds": elapsed,
            "hmm_elapsed_seconds": float(sum(item.elapsed_seconds for item in frozen)),
            "peak_rss_gib": peak_rss_gib(),
            "cpu_only": True,
            "numba_threads": int(get_nested(config, "runtime.numba_num_threads")),
            "versions": runtime_versions(),
        },
        "submission": {
            "file": str(submission_path()),
            "rows": len(submission),
            "sha256": sha256_file(submission_path()),
            "logical_sha256": logical_frame_sha256(submission),
            "minimum_tvt": float(submission["tvt"].min()),
            "maximum_tvt": float(submission["tvt"].max()),
            "mean_tvt": float(submission["tvt"].mean()),
            "std_tvt": float(submission["tvt"].std()),
            "competition_submission": "not_approved_not_started",
        },
        "artifacts": {
            "prediction": prediction_record,
            "segment": segment_record,
            "exp226_kappa": kappa_record,
            "input_manifest": input_record,
            "decoder_manifest": decoder_record,
        },
    }
    summary_record = write_json(
        out_dir / f"{EXPERIMENT_NAME}_inference_summary.json",
        summary,
    )
    summary["artifacts"]["summary"] = summary_record
    write_json(
        metrics_path(),
        {
            "experiment": EXPERIMENT_NAME,
            "status": summary["status"],
            "cv": float(get_nested(config, "stage_1_result.candidate_rmse_ft")),
            "public_lb": None,
            "private_lb": None,
            "inference": summary,
        },
    )
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True), flush=True)
    print(submission.head(20).to_string(index=False), flush=True)
    return summary


# %% [markdown]
# ## 7. Setup and input preflight

# %%
CONFIG = load_config()
EXECUTION_PREVIEW = validate_inference_execution(CONFIG)
SCIENTIFIC_PREVIEW = build_frozen_oof_scientific_contract(CONFIG)
print(
    json.dumps(
        {
            "experiment": EXPERIMENT_NAME,
            "route": get_nested(CONFIG, "experiment.route"),
            "stage_1_all_pass": get_nested(CONFIG, "implementation.stage_1_all_pass"),
            "execution": EXECUTION_PREVIEW,
            "scientific_contract_sha256": hashlib.sha256(stable_json_bytes(SCIENTIFIC_PREVIEW)).hexdigest(),
            "selected_candidate": get_nested(CONFIG, "inference.selected_candidate"),
            "competition_submission_approved": get_nested(CONFIG, "execution.competition_submission_approved"),
        },
        indent=2,
        sort_keys=True,
    ),
    flush=True,
)

# %% [markdown]
# ## 8. Generate the three-well current-test candidate

# %%
if __name__ == "__main__":
    INFERENCE_SUMMARY = run_inference(CONFIG)

# %% [markdown]
# ## 9. Submission, technical gates, SHA, and summary
#
# The execution cell prints the complete technical gate, submission summary,
# SHA records, and the first 20 rows. Kaggle competition submission is not
# performed by this notebook.
'''


def extracted_blocks(source: str) -> dict[str, str]:
    tree = ast.parse(source)
    lines = source.splitlines()
    blocks: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if node.name not in EXTRACT_NAMES:
            continue
        start = node.lineno
        decorators = getattr(node, "decorator_list", [])
        if decorators:
            start = min(start, *(decorator.lineno for decorator in decorators))
        if node.end_lineno is None:
            raise RuntimeError(f"missing end_lineno for {node.name}")
        blocks[node.name] = "\n".join(lines[start - 1 : node.end_lineno])
    missing = sorted(set(EXTRACT_NAMES) - set(blocks))
    if missing:
        raise RuntimeError(f"missing train-source blocks: {missing}")
    return blocks


def section(title: str, blocks: dict[str, str], names: tuple[str, ...]) -> str:
    body = "\n\n\n".join(blocks[name] for name in names)
    return f"\n# %% [markdown]\n# ## {title}\n\n# %%\n{body}\n"


def main() -> None:
    source = TRAIN_SOURCE.read_text(encoding="utf-8")
    blocks = extracted_blocks(source)
    output = HEADER
    output += "\n\n" + blocks["get_nested"] + "\n"
    output += HELPERS
    output += section(
        "2A. Shared deterministic SHA and output helpers",
        blocks,
        (
            "to_jsonable",
            "stable_json_bytes",
            "sha256_file",
            "logical_frame_sha256",
            "array_bundle_sha256",
            "write_json",
            "write_csv",
            "write_deterministic_gzip_csv",
            "peak_rss_gib",
            "runtime_versions",
        ),
    )
    output += EXP226_AND_RAW
    output += section(
        "4A. Frozen K16 and Huber input preparation",
        blocks,
        (
            "huber_log_likelihood",
            "robust_initial_rate",
            "prefix_stats",
            "k16_segment_half_life",
            "zero_state_geometry_identity",
            "prepare_hmm_inputs",
        ),
    )
    output += section(
        "5. Frozen geometry-centered exact forward-backward decoder",
        blocks,
        (
            "_hmm2_fb_geometry_mean_reversion",
            "run_geometry_mean_reverting_hmm",
        ),
    )
    output += section(
        "6A. Frozen prediction record helpers",
        blocks,
        (
            "FrozenWell",
            "prediction_frame",
            "segment_contract_frame",
        ),
    )
    output += FREEZE_AND_RUN
    OUTPUT_SOURCE.write_text(dedent(output).lstrip(), encoding="utf-8")
    print(OUTPUT_SOURCE.relative_to(ROOT))


if __name__ == "__main__":
    main()
