# %% [markdown]
# # exp490 geometry-centered mean-reverting offset HMM inference
#
# This hidden-dynamic version 2 fixes the failed submission ref 55163886 without
# promoting or altering exp490. Before expensive work it scans the complete
# mounted sample/horizontal/typewell inventory, derives the runtime row/well
# contract, regenerates the SHA-pinned exp226 geometry-only field, and applies
# exactly the frozen exp490 mean-reverting Huber HMM. Competition submission is
# outside this notebook and a Kaggle run requires separate approval.

# %% [markdown]
# ## Contents
# 1. Imports and frozen execution contract
# 2. Notebook-safe paths, SHA, and scientific-contract helpers
# 3. Trusted exp226 geometry-only runtime-test regeneration
# 4. Raw-test input and K16 half-life helpers
# 5. Frozen geometry-centered exact forward-backward decoder
# 6. Per-well prediction and generated-artifact helpers
# 7. Setup and input preflight
# 8. Generate the runtime-sized hidden-dynamic candidate
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


def get_nested(
    mapping: Mapping[str, Any],
    dotted_key: str,
    default: Any = None,
) -> Any:
    value: Any = mapping
    for key in dotted_key.split("."):
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value

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


def validate_runtime_test_inventory(
    sample: pd.DataFrame,
    test_dir: Path,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Validate the complete mounted test inventory before expensive inference."""
    identity = parse_sample_identity(sample)
    if identity.empty:
        raise ValueError("sample submission is empty")
    sample_wells = sorted(identity["well"].unique().tolist())
    horizontal_files = sorted(test_dir.glob("*__horizontal_well.csv"))
    typewell_files = sorted(test_dir.glob("*__typewell.csv"))
    horizontal_by_well = {
        path.name.removesuffix("__horizontal_well.csv"): path
        for path in horizontal_files
    }
    typewell_by_well = {
        path.name.removesuffix("__typewell.csv"): path
        for path in typewell_files
    }
    if len(horizontal_by_well) != len(horizontal_files):
        raise ValueError("duplicate horizontal test well files")
    if len(typewell_by_well) != len(typewell_files):
        raise ValueError("duplicate typewell test well files")
    if sorted(horizontal_by_well) != sample_wells:
        raise ValueError("sample and horizontal test well sets differ")
    if sorted(typewell_by_well) != sample_wells:
        raise ValueError("sample and typewell test well sets differ")

    rows_by_well: dict[str, int] = {}
    for well in sample_wells:
        tvt_input = pd.read_csv(
            horizontal_by_well[well],
            usecols=["TVT_input"],
        )["TVT_input"]
        unknown_rows = np.flatnonzero(tvt_input.isna().to_numpy()).astype(np.int64)
        sample_rows = np.sort(
            identity.loc[identity["well"] == well, "row_idx"].to_numpy(np.int64)
        )
        if not np.array_equal(sample_rows, unknown_rows):
            raise ValueError(f"{well}: sample ids and raw unknown-suffix rows differ")
        if len(unknown_rows) == 0:
            raise ValueError(f"{well}: test well has no unknown-suffix rows")
        rows_by_well[well] = int(len(unknown_rows))

    if sum(rows_by_well.values()) != len(identity):
        raise ValueError("sample and raw unknown-suffix row totals differ")
    return identity, sample_wells, {
        "policy": "full_mounted_test_inventory_before_exp226_fit",
        "rows": int(len(identity)),
        "wells": int(len(sample_wells)),
        "well_ids": sample_wells,
        "unknown_rows_by_well": rows_by_well,
        "sample_raw_identity_exact": True,
        "horizontal_typewell_well_sets_exact": True,
    }


def validate_inference_execution(
    config: Mapping[str, Any],
    *,
    require_run_approval: bool = True,
) -> dict[str, Any]:
    expected = {
        "scientific_variants": 1,
        "exp226_full_train_fits": 1,
        "exp226_test_geometry_well_runs": "runtime_sample_well_count",
        "candidate_hmm_well_runs": "runtime_sample_well_count",
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "pf_runs": 0,
        "beam_runs": 0,
        "gpu_runs": 0,
    }
    observed = dict(get_nested(config, "execution_contract.hidden_dynamic_inference_v2"))
    if observed != expected:
        raise ValueError(f"inference execution contract changed: {observed}")
    checks = {
        "run_inference": bool(get_nested(config, "execution.run_inference")),
        "create_submission": bool(get_nested(config, "execution.create_submission")),
        "inference_enabled": bool(get_nested(config, "inference.enabled")),
        "preserve_fail_close": bool(get_nested(config, "inference.preserve_stage_1_fail_close")),
        "stage_1_failed": not bool(get_nested(config, "implementation.stage_1_all_pass")),
        "competition_submission_disabled": not bool(
            get_nested(config, "execution.competition_submission_approved")
        ),
        "gpu_disabled": not bool(get_nested(config, "runtime.kaggle.enable_gpu")),
        "internet_disabled": not bool(get_nested(config, "runtime.kaggle.enable_internet")),
    }
    if require_run_approval:
        checks["hidden_dynamic_v2_run_approved"] = bool(
            get_nested(config, "execution.hidden_dynamic_v2_run_approved")
        )
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"inference authorization contract failed: {failed}")
    return {**expected, "execution_basis": "hidden_dynamic_fix_after_submission_ref_55163886"}


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

# %% [markdown]
# ## 2A. Shared deterministic SHA and output helpers

# %%
def to_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        to_jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_frame_sha256(frame: pd.DataFrame) -> str:
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, lineterminator="\n")
    return hashlib.sha256(buffer.getvalue().encode("utf-8")).hexdigest()


def array_bundle_sha256(**arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        to_jsonable(payload),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    path.write_bytes(data + b"\n")
    return {"path": str(path), "sha256": sha256_file(path)}


def write_csv(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    return {
        "path": str(path),
        "rows": len(frame),
        "sha256": sha256_file(path),
        "logical_sha256": logical_frame_sha256(frame),
    }


def write_deterministic_gzip_csv(
    path: Path,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_bytes = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    with path.open("wb") as raw:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw,
            compresslevel=6,
            mtime=0,
        ) as compressed:
            compressed.write(csv_bytes)
    return {
        "path": str(path),
        "rows": len(frame),
        "raw_gzip_sha256": sha256_file(path),
        "decompressed_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "logical_sha256": logical_frame_sha256(frame),
    }


def peak_rss_gib() -> float:
    rss_kib = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if platform.system() == "Darwin":
        return rss_kib / (1024.0**3)
    return rss_kib / (1024.0**2)


def runtime_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyyaml": yaml.__version__,
    }
    try:
        import numba

        versions["numba"] = numba.__version__
    except Exception:
        versions["numba"] = "unavailable"
    return versions

# %% [markdown]
# ## 3. Trusted exp226 geometry-only runtime-test regeneration

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

# %% [markdown]
# ## 4A. Frozen K16 and Huber input preparation

# %%
def huber_log_likelihood(zscore: np.ndarray, delta: float) -> np.ndarray:
    z = np.asarray(zscore, dtype=np.float64)
    abs_z = np.abs(z)
    loss = np.where(
        abs_z <= float(delta),
        0.5 * z * z,
        float(delta) * abs_z - 0.5 * float(delta) ** 2,
    )
    return -loss


def robust_initial_rate(
    known_prefix: pd.DataFrame,
    window_rows: int = 30,
    *,
    min_valid_steps: int = 3,
    fallback_rate: float = 0.0,
) -> tuple[float, int, int]:
    tail = known_prefix.tail(int(window_rows))
    tvt = pd.to_numeric(tail["TVT_input"], errors="coerce").to_numpy(np.float64)
    z = pd.to_numeric(tail["Z"], errors="coerce").to_numpy(np.float64)
    md = pd.to_numeric(tail["MD"], errors="coerce").to_numpy(np.float64)
    dtvt = np.diff(tvt)
    dz = np.diff(z)
    dmd = np.diff(md)
    valid = np.isfinite(dtvt) & np.isfinite(dz) & np.isfinite(dmd) & (dmd > 0.0)
    valid_steps = int(valid.sum())
    if valid_steps < int(min_valid_steps):
        return float(fallback_rate), int(len(tail)), valid_steps
    rate = float(np.median((dtvt[valid] + dz[valid]) / dmd[valid]))
    if not np.isfinite(rate):
        rate = float(fallback_rate)
    return rate, int(len(tail)), valid_steps


def prefix_stats(
    horizontal: pd.DataFrame,
    typewell_tvt: np.ndarray,
    typewell_gr: np.ndarray,
) -> tuple[float, float, int, int]:
    known = horizontal.loc[horizontal["TVT_input"].notna()]
    known_gr = (
        pd.to_numeric(known["GR"], errors="coerce")
        .fillna(0.0)
        .to_numpy(np.float64)
    )
    known_tvt = pd.to_numeric(
        known["TVT_input"], errors="raise"
    ).to_numpy(np.float64)
    typewell_at_known = np.interp(known_tvt, typewell_tvt, typewell_gr)
    residual = known_gr - typewell_at_known
    sigma = float(np.clip(np.std(residual), 10.0, 60.0))
    if not np.isfinite(sigma):
        raise ValueError("known-prefix GR residual sigma is not finite")
    initial_rate, effective_rows, valid_steps = robust_initial_rate(known, 30)
    return sigma, initial_rate, effective_rows, valid_steps


def k16_segment_half_life(
    unknown_md: np.ndarray,
    *,
    last_known_md: float,
    segment_count: int = 16,
) -> dict[str, np.ndarray]:
    md = np.asarray(unknown_md, dtype=np.float64)
    if md.ndim != 1 or len(md) < int(segment_count):
        raise ValueError("K16 requires at least one destination row per segment")
    if not np.isfinite(md).all() or not np.isfinite(last_known_md):
        raise ValueError("K16 MD values must be finite")
    dmd = np.diff(np.concatenate([[float(last_known_md)], md]))
    if not np.all(dmd > 0.0):
        raise ValueError("every transition-entering dMD must be strictly positive")
    edges = np.linspace(0.0, float(len(md)), int(segment_count) + 1)
    destination_rows = np.arange(1, len(md) + 1, dtype=np.float64)
    segment_id = np.clip(
        np.searchsorted(edges[1:], destination_rows, side="left"),
        0,
        int(segment_count) - 1,
    ).astype(np.int16)
    segment_span = np.bincount(
        segment_id.astype(np.int64),
        weights=dmd,
        minlength=int(segment_count),
    ).astype(np.float64)
    segment_rows = np.bincount(
        segment_id.astype(np.int64),
        minlength=int(segment_count),
    ).astype(np.int64)
    if np.any(segment_rows <= 0) or np.any(segment_span <= 0.0):
        raise ValueError("every K16 destination segment must be non-empty and positive")
    rho = np.power(2.0, -dmd / segment_span[segment_id])
    if not np.isfinite(rho).all() or np.any(rho <= 0.0) or np.any(rho > 1.0):
        raise ValueError("rho must be finite and in (0, 1]")
    cumulative = np.asarray(
        [
            np.prod(rho[segment_id == segment])
            for segment in range(int(segment_count))
        ],
        dtype=np.float64,
    )
    return {
        "dmd": dmd,
        "segment_id": segment_id,
        "segment_span": segment_span,
        "segment_rows": segment_rows,
        "rho": rho,
        "segment_cumulative_rho": cumulative,
        "edges": edges,
    }


def zero_state_geometry_identity(
    dmd: np.ndarray,
    rho: np.ndarray,
) -> dict[str, Any]:
    dmd_values = np.asarray(dmd, dtype=np.float64)
    rho_values = np.asarray(rho, dtype=np.float64)
    rate_center = 0.998 * rho_values * np.zeros_like(dmd_values)
    offset_center = rho_values * np.zeros_like(dmd_values) + rate_center * dmd_values
    maximum_abs_offset = float(np.max(np.abs(offset_center)))
    return {
        "pass": bool(maximum_abs_offset == 0.0),
        "maximum_abs_offset_ft": maximum_abs_offset,
        "rows": len(dmd_values),
    }


def prepare_hmm_inputs(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    geop_tvt: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    if "TVT" in horizontal.columns:
        raise ValueError("candidate HMM preparation forbids horizontal truth")
    required_horizontal = {"MD", "Z", "GR", "TVT_input"}
    if not required_horizontal.issubset(horizontal.columns):
        missing = sorted(required_horizontal - set(horizontal.columns))
        raise ValueError(f"horizontal input is missing {missing}")
    typewell_tvt = pd.to_numeric(
        typewell["TVT"], errors="raise"
    ).to_numpy(np.float64)
    typewell_gr = (
        pd.to_numeric(typewell["GR"], errors="coerce")
        .ffill()
        .bfill()
        .to_numpy(np.float64)
    )
    if (
        len(typewell_tvt) < 2
        or not np.isfinite(typewell_tvt).all()
        or not np.isfinite(typewell_gr).all()
    ):
        raise ValueError("typewell TVT/GR must contain finite interpolation support")
    known = horizontal.loc[horizontal["TVT_input"].notna()]
    evaluation = horizontal.loc[horizontal["TVT_input"].isna()]
    if len(known) < 4 or len(evaluation) == 0:
        raise ValueError("well needs four known rows and a non-empty unknown suffix")
    geop = np.asarray(geop_tvt, dtype=np.float64)
    if len(geop) != len(evaluation) or not np.isfinite(geop).all():
        raise ValueError("exp226 tvt_geop must align and be finite")

    sigma, initial_rate_diagnostic, rate_rows, valid_steps = prefix_stats(
        horizontal,
        typewell_tvt,
        typewell_gr,
    )
    last_known = known.iloc[-1]
    md = pd.to_numeric(evaluation["MD"], errors="raise").to_numpy(np.float64)
    segment = k16_segment_half_life(
        md,
        last_known_md=float(last_known["MD"]),
        segment_count=int(get_nested(config, "model.k16_segment.count")),
    )
    gr_fill = float(np.mean(typewell_gr))
    gr = (
        pd.to_numeric(horizontal["GR"], errors="coerce")
        .interpolate(limit_direction="both")
        .fillna(gr_fill)
        .to_numpy(np.float64)[evaluation.index]
    )
    hmm = get_nested(config, "model.hmm")
    step = float(hmm["step_ft"])
    grid = np.arange(
        float(hmm["delta_min_ft"]),
        float(hmm["delta_max_ft"]) + 0.5 * step,
        step,
        dtype=np.float64,
    )
    rates = np.linspace(
        -float(hmm["rate_span"]),
        float(hmm["rate_span"]),
        int(hmm["n_rates"]),
        dtype=np.float64,
    )
    absolute_tvt_states = geop[:, None] + grid[None, :]
    gr_grid = np.interp(absolute_tvt_states, typewell_tvt, typewell_gr)
    zscore = (gr[:, None] - gr_grid) / sigma
    emission = get_nested(config, "model.emission")
    emission_ll = huber_log_likelihood(
        zscore,
        float(emission["huber_delta"]),
    ).astype(np.float32)
    if not np.isfinite(emission_ll).all():
        raise ValueError("Huber emission contains non-finite values")
    start_delta = float(hmm["start_delta_ft"])
    native_typewell = (absolute_tvt_states >= float(typewell_tvt.min())) & (
        absolute_tvt_states <= float(typewell_tvt.max())
    )
    return {
        "emission_ll": emission_ll,
        "grid": grid,
        "rates": rates,
        "dmd": segment["dmd"],
        "segment_id": segment["segment_id"],
        "segment_span": segment["segment_span"],
        "segment_rows": segment["segment_rows"],
        "rho": segment["rho"],
        "segment_cumulative_rho": segment["segment_cumulative_rho"],
        "segment_edges": segment["edges"],
        "eval_index": evaluation.index.to_numpy(np.int64),
        "start_p": float((start_delta - grid[0]) / step),
        "prefix_rows": int(len(known)),
        "prefix_sigma": sigma,
        "prefix_initial_rate_diagnostic_only": initial_rate_diagnostic,
        "initial_rate_effective_rows": int(rate_rows),
        "initial_rate_valid_steps": int(valid_steps),
        "emission_finite_coverage": float(np.isfinite(emission_ll).mean()),
        "native_typewell_state_coverage": float(native_typewell.mean()),
        "zero_quantization_error_ft": float(np.min(np.abs(grid - start_delta))),
        "zero_state_identity": zero_state_geometry_identity(
            segment["dmd"],
            segment["rho"],
        ),
    }

# %% [markdown]
# ## 5. Frozen geometry-centered exact forward-backward decoder

# %%
@njit(cache=True, nogil=True, parallel=True)
def _hmm2_fb_geometry_mean_reversion(
    emission_ll,
    dmd,
    rho,
    step,
    rates,
    sig_r,
    sig_p,
    start_p,
    start_sig,
    initial_rate,
    initial_rate_sig,
    lam,
    momentum,
):
    time_count, position_count = emission_ll.shape
    rate_count = len(rates)
    rate_step = rates[1] - rates[0]
    zero_position = 80.0 / step
    negative = np.float32(-1.0e18)
    alpha = np.full(
        (time_count, position_count, rate_count),
        negative,
        np.float32,
    )
    previous = np.full((position_count, rate_count), negative, np.float32)
    for position_index in range(position_count):
        delta_position = (position_index - start_p) * step
        position_prior = -0.5 * (delta_position / start_sig) ** 2
        if position_prior < -60.0:
            continue
        for rate_index in range(rate_count):
            delta_rate = (rates[rate_index] - initial_rate) / initial_rate_sig
            previous[position_index, rate_index] = np.float32(
                position_prior - 0.5 * delta_rate * delta_rate
            )

    after_rate = np.empty((position_count, rate_count), np.float32)
    current = np.empty((position_count, rate_count), np.float32)
    for time_index in range(time_count):
        row_rho = rho[time_index]
        sigma_rate_step = sig_r * np.sqrt(dmd[time_index])
        rate_variance_cells = (sigma_rate_step / rate_step) ** 2

        for position_index in prange(position_count):
            for destination_rate in range(rate_count):
                best = negative
                source_start = max(destination_rate - 1, 0)
                source_end = min(destination_rate + 1, rate_count - 1)
                for source_rate in range(source_start, source_end + 1):
                    mean_move = (
                        momentum * row_rho * rates[source_rate]
                        - rates[source_rate]
                    ) / rate_step
                    probability_plus = max(
                        0.5 * (rate_variance_cells + mean_move),
                        1.0e-12,
                    )
                    probability_minus = max(
                        0.5 * (rate_variance_cells - mean_move),
                        1.0e-12,
                    )
                    probability_total = probability_plus + probability_minus
                    if probability_total > 0.9:
                        probability_plus *= 0.9 / probability_total
                        probability_minus *= 0.9 / probability_total
                    move = destination_rate - source_rate
                    if move == -1:
                        log_probability = np.log(probability_minus)
                    elif move == 0:
                        log_probability = np.log(
                            1.0 - probability_plus - probability_minus
                        )
                    else:
                        log_probability = np.log(probability_plus)
                    value = (
                        previous[position_index, source_rate] + log_probability
                    )
                    if value > best:
                        best = value
                if best > negative / 2:
                    total = 0.0
                    for source_rate in range(source_start, source_end + 1):
                        mean_move = (
                            momentum * row_rho * rates[source_rate]
                            - rates[source_rate]
                        ) / rate_step
                        probability_plus = max(
                            0.5 * (rate_variance_cells + mean_move),
                            1.0e-12,
                        )
                        probability_minus = max(
                            0.5 * (rate_variance_cells - mean_move),
                            1.0e-12,
                        )
                        probability_total = probability_plus + probability_minus
                        if probability_total > 0.9:
                            probability_plus *= 0.9 / probability_total
                            probability_minus *= 0.9 / probability_total
                        move = destination_rate - source_rate
                        if move == -1:
                            log_probability = np.log(probability_minus)
                        elif move == 0:
                            log_probability = np.log(
                                1.0 - probability_plus - probability_minus
                            )
                        else:
                            log_probability = np.log(probability_plus)
                        total += np.exp(
                            previous[position_index, source_rate]
                            + log_probability
                            - best
                        )
                    after_rate[position_index, destination_rate] = np.float32(
                        best + np.log(total)
                    )
                else:
                    after_rate[position_index, destination_rate] = negative

        sigma_position = max(sig_p, 0.35 * step)
        for destination_rate in prange(rate_count):
            rate_displacement_cells = (
                rates[destination_rate] * dmd[time_index] / step
            )
            for destination_position in range(position_count):
                lower_source = zero_position + (
                    destination_position
                    - 2.5
                    - zero_position
                    - rate_displacement_cells
                ) / row_rho
                upper_source = zero_position + (
                    destination_position
                    + 2.5
                    - zero_position
                    - rate_displacement_cells
                ) / row_rho
                source_start = max(int(np.ceil(lower_source)), 0)
                source_end = min(int(np.ceil(upper_source)) - 1, position_count - 1)
                best = negative
                for source_position in range(source_start, source_end + 1):
                    center = (
                        zero_position
                        + row_rho * (source_position - zero_position)
                        + rate_displacement_cells
                    )
                    base = int(np.floor(center + 0.5))
                    kernel_index = destination_position - (base - 2)
                    if kernel_index < 0 or kernel_index >= 5:
                        continue
                    kernel = np.empty(5)
                    for offset in range(5):
                        residual = (base - 2 + offset - center) * step
                        kernel[offset] = -0.5 * (
                            residual / sigma_position
                        ) ** 2
                    kernel_max = np.max(kernel)
                    log_normalizer = kernel_max + np.log(
                        np.sum(np.exp(kernel - kernel_max))
                    )
                    value = (
                        after_rate[source_position, destination_rate]
                        + kernel[kernel_index]
                        - log_normalizer
                    )
                    if value > best:
                        best = value
                if best > negative / 2:
                    total = 0.0
                    for source_position in range(source_start, source_end + 1):
                        center = (
                            zero_position
                            + row_rho * (source_position - zero_position)
                            + rate_displacement_cells
                        )
                        base = int(np.floor(center + 0.5))
                        kernel_index = destination_position - (base - 2)
                        if kernel_index < 0 or kernel_index >= 5:
                            continue
                        kernel = np.empty(5)
                        for offset in range(5):
                            residual = (base - 2 + offset - center) * step
                            kernel[offset] = -0.5 * (
                                residual / sigma_position
                            ) ** 2
                        kernel_max = np.max(kernel)
                        log_normalizer = kernel_max + np.log(
                            np.sum(np.exp(kernel - kernel_max))
                        )
                        total += np.exp(
                            after_rate[source_position, destination_rate]
                            + kernel[kernel_index]
                            - log_normalizer
                            - best
                        )
                    current[destination_position, destination_rate] = np.float32(
                        best
                        + np.log(total)
                        + lam * emission_ll[time_index, destination_position]
                    )
                else:
                    current[destination_position, destination_rate] = negative

        for position_index in range(position_count):
            for rate_index in range(rate_count):
                alpha[time_index, position_index, rate_index] = current[
                    position_index, rate_index
                ]
                previous[position_index, rate_index] = current[
                    position_index, rate_index
                ]

    last_best = np.max(alpha[time_count - 1])
    last_total = np.sum(np.exp(alpha[time_count - 1] - last_best))
    log_likelihood = float(last_best) + np.log(last_total)
    position_posterior = np.zeros((time_count, position_count), np.float64)
    beta_next = np.zeros((position_count, rate_count), np.float32)
    values = alpha[time_count - 1] + beta_next
    best = np.max(values)
    total = 0.0
    for position_index in range(position_count):
        accumulator = 0.0
        for rate_index in range(rate_count):
            accumulator += np.exp(values[position_index, rate_index] - best)
        position_posterior[time_count - 1, position_index] = accumulator
        total += accumulator
    position_posterior[time_count - 1] /= total

    beta_current = np.empty((position_count, rate_count), np.float32)
    beta_after_position = np.empty((position_count, rate_count), np.float32)
    for time_index in range(time_count - 1, 0, -1):
        row_rho = rho[time_index]
        sigma_position = max(sig_p, 0.35 * step)
        for source_position in prange(position_count):
            for destination_rate in range(rate_count):
                rate_displacement_cells = (
                    rates[destination_rate] * dmd[time_index] / step
                )
                center = (
                    zero_position
                    + row_rho * (source_position - zero_position)
                    + rate_displacement_cells
                )
                base = int(np.floor(center + 0.5))
                kernel = np.empty(5)
                for offset in range(5):
                    residual = (base - 2 + offset - center) * step
                    kernel[offset] = -0.5 * (residual / sigma_position) ** 2
                kernel_max = np.max(kernel)
                log_normalizer = kernel_max + np.log(
                    np.sum(np.exp(kernel - kernel_max))
                )
                best = negative
                for offset in range(5):
                    destination_position = base - 2 + offset
                    if 0 <= destination_position < position_count:
                        value = (
                            kernel[offset]
                            - log_normalizer
                            + lam
                            * emission_ll[time_index, destination_position]
                            + beta_next[destination_position, destination_rate]
                        )
                        if value > best:
                            best = value
                if best > negative / 2:
                    total = 0.0
                    for offset in range(5):
                        destination_position = base - 2 + offset
                        if 0 <= destination_position < position_count:
                            total += np.exp(
                                kernel[offset]
                                - log_normalizer
                                + lam
                                * emission_ll[time_index, destination_position]
                                + beta_next[destination_position, destination_rate]
                                - best
                            )
                    beta_after_position[
                        source_position, destination_rate
                    ] = np.float32(best + np.log(total))
                else:
                    beta_after_position[
                        source_position, destination_rate
                    ] = negative

        sigma_rate_step = sig_r * np.sqrt(dmd[time_index])
        rate_variance_cells = (sigma_rate_step / rate_step) ** 2
        for source_position in prange(position_count):
            for source_rate in range(rate_count):
                mean_move = (
                    momentum * row_rho * rates[source_rate]
                    - rates[source_rate]
                ) / rate_step
                probability_plus = max(
                    0.5 * (rate_variance_cells + mean_move),
                    1.0e-12,
                )
                probability_minus = max(
                    0.5 * (rate_variance_cells - mean_move),
                    1.0e-12,
                )
                probability_total = probability_plus + probability_minus
                if probability_total > 0.9:
                    probability_plus *= 0.9 / probability_total
                    probability_minus *= 0.9 / probability_total
                best = negative
                destination_start = max(source_rate - 1, 0)
                destination_end = min(source_rate + 1, rate_count - 1)
                for destination_rate in range(
                    destination_start, destination_end + 1
                ):
                    move = destination_rate - source_rate
                    if move == -1:
                        log_probability = np.log(probability_minus)
                    elif move == 0:
                        log_probability = np.log(
                            1.0 - probability_plus - probability_minus
                        )
                    else:
                        log_probability = np.log(probability_plus)
                    value = (
                        log_probability
                        + beta_after_position[source_position, destination_rate]
                    )
                    if value > best:
                        best = value
                if best > negative / 2:
                    total = 0.0
                    for destination_rate in range(
                        destination_start, destination_end + 1
                    ):
                        move = destination_rate - source_rate
                        if move == -1:
                            log_probability = np.log(probability_minus)
                        elif move == 0:
                            log_probability = np.log(
                                1.0 - probability_plus - probability_minus
                            )
                        else:
                            log_probability = np.log(probability_plus)
                        total += np.exp(
                            log_probability
                            + beta_after_position[
                                source_position, destination_rate
                            ]
                            - best
                        )
                    beta_current[source_position, source_rate] = np.float32(
                        best + np.log(total)
                    )
                else:
                    beta_current[source_position, source_rate] = negative

        values = alpha[time_index - 1] + beta_current
        best = np.max(values)
        total = 0.0
        for position_index in range(position_count):
            accumulator = 0.0
            for rate_index in range(rate_count):
                accumulator += np.exp(values[position_index, rate_index] - best)
            position_posterior[time_index - 1, position_index] = accumulator
            total += accumulator
        position_posterior[time_index - 1] /= total
        for position_index in range(position_count):
            for rate_index in range(rate_count):
                beta_next[position_index, rate_index] = beta_current[
                    position_index, rate_index
                ]

    maximum_normalization_error = 0.0
    for time_index in range(time_count):
        row_error = abs(np.sum(position_posterior[time_index]) - 1.0)
        if row_error > maximum_normalization_error:
            maximum_normalization_error = row_error
    return position_posterior, log_likelihood, maximum_normalization_error


def run_geometry_mean_reverting_hmm(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    geop_tvt: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    prepared = prepare_hmm_inputs(horizontal, typewell, geop_tvt, config)
    hmm = get_nested(config, "model.hmm")
    emission = get_nested(config, "model.emission")
    posterior, log_likelihood, normalization_error = (
        _hmm2_fb_geometry_mean_reversion(
            prepared["emission_ll"],
            prepared["dmd"].astype(np.float64),
            prepared["rho"].astype(np.float64),
            float(hmm["step_ft"]),
            prepared["rates"].astype(np.float64),
            float(hmm["sig_r"]),
            float(hmm["sig_p"]),
            float(prepared["start_p"]),
            float(hmm["start_sig_ft"]),
            float(hmm["initial_rate"]),
            float(hmm["initial_rate_sig"]),
            float(emission["lam"]),
            float(hmm["mom"]),
        )
    )
    grid = prepared["grid"]
    delta_mean = posterior @ grid
    variance = posterior @ (grid**2) - delta_mean**2
    standard_deviation = np.sqrt(np.maximum(variance, 0.0))
    del posterior
    gc.collect()
    return {
        **prepared,
        "delta_mean": np.asarray(delta_mean, dtype=np.float64),
        "mean": np.asarray(geop_tvt, dtype=np.float64)
        + np.asarray(delta_mean, dtype=np.float64),
        "std": np.asarray(standard_deviation, dtype=np.float64),
        "log_likelihood": float(log_likelihood),
        "maximum_posterior_normalization_error": float(normalization_error),
    }

# %% [markdown]
# ## 6A. Frozen prediction record helpers

# %%
@dataclass
class FrozenWell:
    well: str
    row_idx: np.ndarray
    suffix_offset: np.ndarray
    tvt_geop: np.ndarray
    prediction: np.ndarray
    delta_mean: np.ndarray
    posterior_std: np.ndarray
    dmd: np.ndarray
    segment_id: np.ndarray
    segment_span: np.ndarray
    segment_rows: np.ndarray
    segment_cumulative_rho: np.ndarray
    rho: np.ndarray
    prefix_rows: int
    prefix_sigma: float
    maximum_posterior_normalization_error: float
    zero_state_identity_pass: bool
    emission_finite_coverage: float
    elapsed_seconds: float
    prediction_sha256: str
    raw_input_sha256: dict[str, str]


def prediction_frame(frozen_wells: Sequence[FrozenWell]) -> pd.DataFrame:
    pieces = []
    for item in frozen_wells:
        pieces.append(
            pd.DataFrame(
                {
                    "id": [
                        f"{item.well}_{int(row)}" for row in item.row_idx
                    ],
                    "well": item.well,
                    "row_idx": item.row_idx,
                    "suffix_offset": item.suffix_offset,
                    "tvt_geop": item.tvt_geop,
                    "geometry_mean_reverting_hmm": item.prediction,
                    "geometry_mean_reverting_delta_mean": item.delta_mean,
                    "geometry_mean_reverting_hmm_std": item.posterior_std,
                    "dmd": item.dmd,
                    "k16_segment_id": item.segment_id,
                    "k16_segment_span": item.segment_span[item.segment_id],
                    "rho": item.rho,
                }
            )
        )
    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["well", "row_idx"], kind="mergesort")
        .reset_index(drop=True)
    )


def segment_contract_frame(
    frozen_wells: Sequence[FrozenWell],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in frozen_wells:
        for segment in range(len(item.segment_span)):
            mask = item.segment_id == segment
            rows.append(
                {
                    "well": item.well,
                    "k16_segment_id": segment,
                    "destination_start_suffix_offset": int(
                        item.suffix_offset[mask][0]
                    ),
                    "destination_end_suffix_offset_exclusive": int(
                        item.suffix_offset[mask][-1] + 1
                    ),
                    "rows": int(mask.sum()),
                    "dmd_span": float(item.segment_span[segment]),
                    "rho_product": float(
                        item.segment_cumulative_rho[segment]
                    ),
                    "rho_product_abs_error_vs_half": float(
                        abs(item.segment_cumulative_rho[segment] - 0.5)
                    ),
                    "minimum_rho": float(np.min(item.rho[mask])),
                    "maximum_rho": float(np.max(item.rho[mask])),
                    "positive_dmd": bool(np.all(item.dmd[mask] > 0.0)),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["well", "k16_segment_id"], kind="mergesort"
    )

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
    sample_sha = sha256_file(sample_path)
    sample = pd.read_csv(sample_path, dtype={"id": str})
    identity, wells, runtime_test_contract = validate_runtime_test_inventory(
        sample,
        data_root / "test",
    )
    runtime_rows = int(len(sample))
    runtime_wells = int(len(wells))
    public_reference = dict(get_nested(config, "data.public_test_reference"))

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
    if len(prediction) != runtime_rows or prediction["well"].nunique() != runtime_wells:
        raise ValueError("prediction row/well coverage changed")
    max_norm = float(max(item.maximum_posterior_normalization_error for item in frozen))
    max_half_life_error = float(segment["rho_product_abs_error_vs_half"].max())
    technical = {
        "exp226_source_sha": exp226_audit["source_sha256"] == str(get_nested(config, "data.exp226_inference_source.source_sha256")),
        "exp226_config_sha": exp226_audit["config_sha256"] == str(get_nested(config, "data.exp226_inference_source.config_sha256")),
        "scientific_contract_parity": scientific_sha == EXPECTED_OOF_SCIENTIFIC_CONTRACT_SHA256,
        "runtime_test_inventory": bool(runtime_test_contract["sample_raw_identity_exact"]),
        "sample_dynamic_row_and_well_coverage": len(prediction) == runtime_rows and prediction["well"].nunique() == runtime_wells,
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
        "sample_submission_sha256": sample_sha,
        "sample_rows": len(sample),
        "test_wells": wells,
        "runtime_test_contract": runtime_test_contract,
        "public_test_reference": {
            **public_reference,
            "sample_sha_matches": sample_sha == str(public_reference["sample_submission_sha256"]),
            "rows_match": runtime_rows == int(public_reference["submission_rows"]),
            "wells_match": runtime_wells == int(public_reference["test_wells"]),
            "role": "audit_only_not_runtime_gate",
        },
        "exp226": exp226_audit,
        "raw_input_sha256_by_well": {item.well: item.raw_input_sha256 for item in frozen},
    }
    input_record = write_json(
        out_dir / f"{EXPERIMENT_NAME}_inference_input_manifest.json",
        input_manifest,
    )
    decoder_manifest = {
        "experiment": EXPERIMENT_NAME,
        "stage": "hidden_dynamic_inference_v2",
        "execution_basis": "hidden_dynamic_fix_after_submission_ref_55163886",
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
        "status": "hidden_dynamic_inference_complete_pending_submit_check_and_submission_approval",
        "route": "pf_beam",
        "execution_basis": "hidden_dynamic_fix_after_submission_ref_55163886",
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
EXECUTION_PREVIEW = validate_inference_execution(CONFIG, require_run_approval=False)
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
# ## 8. Generate the runtime-sized hidden-dynamic candidate

# %%
if __name__ == "__main__":
    INFERENCE_SUMMARY = run_inference(CONFIG)

# %% [markdown]
# ## 9. Submission, technical gates, SHA, and summary
#
# The execution cell prints the complete technical gate, submission summary,
# SHA records, and the first 20 rows. Kaggle competition submission is not
# performed by this notebook.
