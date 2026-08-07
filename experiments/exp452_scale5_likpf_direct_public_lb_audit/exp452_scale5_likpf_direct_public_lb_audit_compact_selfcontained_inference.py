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
# # exp452 scale-5 LikPF direct Public-LB audit inference
#
# exp417で凍結した`likpf_scale_5_x1p0`だけをraw competition testから再生成し、
# sample submission順の`tvt`へそのまま出力する。500 particles、128 seeds、
# `gs x1.0`、full-suffix log-likelihood、temperature 5を固定し、学習、blend、
# selector、gate、postprocess、fallback、他temperature、算術平均を実行しない。
#
# Kaggle runとsubmission.csv生成はconfigの個別承認flagがそろうまでfail-closeする。
# competition submissionを行うコードは含めない。

# %% [markdown]
# ## Contents
#
# 1. Imports
# 2. Frozen scientific and execution contract
# 3. Runtime, path, SHA, and identity helpers
# 4. Sample submission and public-reference helpers
# 5. Frozen exp073 likelihood-PF kernel
# 6. Scale-5-only raw-test generation helpers
# 7. Parity, submission, and manifest helpers
# 8. Setup and frozen configuration preview
# 9. Generate the one scale-5 candidate
# 10. Public-reference parity and submission generation
# 11. Metrics, SHA, and generated artifacts

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import multiprocessing
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml
from IPython.display import display
from joblib import Parallel, delayed

try:
    from numba import njit
except ModuleNotFoundError:
    if os.environ.get("EXP452_IMPORT_ONLY") != "1":
        raise

    def njit(*args: Any, **kwargs: Any) -> Any:
        """Import-only fallback; executable inference still requires Numba."""
        del kwargs
        if args and callable(args[0]):
            return args[0]

        def decorator(function: Any) -> Any:
            return function

        return decorator


# %% [markdown]
# ## 2. Frozen scientific and execution contract

# %%
EXPERIMENT_NAME = "exp452_scale5_likpf_direct_public_lb_audit"
IMPORT_ONLY_ENV = "EXP452_IMPORT_ONLY"
ACTIVE_VARIANT = "likpf_scale_5_x1p0"
FEATURE_FAMILY = "likpf"
SPLIT_NAME = "test"
SEED_NAMESPACE = "SHA256(likpf::test::<well>)"
BASE_SEED = 42
PARTICLES = 500
SEEDS = 128
TEMPERATURE = 5.0
GR_SCALE_MULTIPLIER = 1.0
EXPECTED_EXP413_SOURCE_SHA256 = (
    "0f6fc81e56556aa6db828584ab2a2e58dde9db9cc4b54d6c12fa60e1c68f1388"
)
EXPECTED_EXP413_CONFIG_SHA256 = (
    "d12e6d74a7f567f0873d5513883b3a7d36d0cd5be5231037e7db12f1a74036a7"
)
EXPECTED_EXP073_PF_SOURCE_SHA256 = (
    "4af212a8a1c83e36cdcc0bc912942a62df1fbc94ca67fd75789171afaa1a647e"
)
EXPECTED_PUBLIC_CANDIDATE_CONTENT_SHA256 = (
    "b713ade7adb5b185dacc941edf19aec324bcd7e075a8e903d33a23f59eb809f3"
)
EXPECTED_PUBLIC_REFERENCE_GZIP_SHA256 = (
    "51ec4d4a5c8f48c99497b6d2f9031dc6ce16c19cf16f565423a144575ef2e832"
)
EXPECTED_PUBLIC_REFERENCE_CONTENT_SHA256 = (
    "51be555c9e787703f93731a42c82ab4b35d5d7823528625eb2b06b7d226200cc"
)


def nested(mapping: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    value: Any = mapping
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
    return value


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_frozen_contract(
    config: Mapping[str, Any],
    *,
    require_execution_approval: bool = False,
) -> dict[str, Any]:
    if nested(config, "experiment.name") != EXPERIMENT_NAME:
        raise ValueError("experiment name differs from the frozen exp452 contract")
    if nested(config, "experiment.route") != "pf_beam":
        raise ValueError("exp452 route must remain pf_beam")
    if list(nested(config, "model.active_variants", [])) != [ACTIVE_VARIANT]:
        raise ValueError("exactly one frozen scale-5 candidate must remain active")

    expected_values = {
        "model.pf.particles": PARTICLES,
        "model.pf.seeds": SEEDS,
        "model.pf.seed_indices": [0, 127],
        "model.pf.typewell_grid_step_ft": 0.2,
        "model.pf.initial_position_spread_ft": 4.5,
        "model.pf.initial_rate_spread": 0.01,
        "model.pf.momentum": 0.998,
        "model.pf.rate_noise": 0.002,
        "model.pf.position_noise": 0.005,
        "model.pf.rough_position": 0.1,
        "model.pf.rough_rate": 0.001,
        "model.pf.resample_threshold_fraction": 0.5,
        "model.pf.emission_clip_z2": 600.0,
        "model.pf.typewell_tvt_pad_ft": 100.0,
        "model.aggregation.temperature": TEMPERATURE,
        "model.aggregation.output_dtype": "float32",
        "model.gr_scale.multiplier": GR_SCALE_MULTIPLIER,
        "model.gr_scale.clip": [10.0, 60.0],
        "model.gr_scale.post_multiplier_clip": None,
        "runtime.device": "cpu",
        "runtime.enable_gpu": False,
        "runtime.enable_internet": False,
    }
    for key, expected in expected_values.items():
        actual = nested(config, key)
        if actual != expected:
            raise ValueError(f"frozen contract mismatch: {key}={actual!r}, expected {expected!r}")

    counts = dict(nested(config, "model.execution_count", {}))
    expected_counts = {
        "scientific_variants": 1,
        "train_runs": 0,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "hmm_runs": 0,
        "beam_runs": 0,
        "parent_control_reruns": 0,
    }
    if counts != expected_counts:
        raise ValueError(f"execution-count contract changed: {counts}")

    source_pins = {
        "exp413_source": nested(config, "data.pinned_source.sha256"),
        "exp413_config": nested(config, "data.pinned_source.config_sha256"),
        "exp073_pf_source": nested(config, "data.pinned_source.pf_source_sha256"),
        "public_candidate": nested(
            config, "data.public_reference.expected_candidate_content_sha256"
        ),
        "public_reference_gzip": nested(
            config, "data.public_reference.expected_file_sha256"
        ),
        "public_reference_content": nested(
            config, "data.public_reference.expected_decompressed_sha256"
        ),
    }
    expected_pins = {
        "exp413_source": EXPECTED_EXP413_SOURCE_SHA256,
        "exp413_config": EXPECTED_EXP413_CONFIG_SHA256,
        "exp073_pf_source": EXPECTED_EXP073_PF_SOURCE_SHA256,
        "public_candidate": EXPECTED_PUBLIC_CANDIDATE_CONTENT_SHA256,
        "public_reference_gzip": EXPECTED_PUBLIC_REFERENCE_GZIP_SHA256,
        "public_reference_content": EXPECTED_PUBLIC_REFERENCE_CONTENT_SHA256,
    }
    if source_pins != expected_pins:
        raise ValueError(f"source/content SHA pins changed: {source_pins}")

    execution = dict(nested(config, "execution", {}))
    if execution.get("selected_candidate") != ACTIVE_VARIANT:
        raise ValueError("execution candidate must remain likpf_scale_5_x1p0")
    if bool(execution.get("run_train")):
        raise ValueError("exp452 cannot run train")
    if bool(execution.get("competition_submission_approved")):
        raise ValueError("competition submission approval is outside this notebook run")
    if bool(nested(config, "implementation.competition_submission_started")):
        raise ValueError("competition submission must not start during implementation")

    if require_execution_approval:
        required = {
            "implementation_approved": execution.get("implementation_approved"),
            "canonical_notebook_adoption_approved": execution.get(
                "canonical_notebook_adoption_approved"
            ),
            "kaggle_run_approved": execution.get("kaggle_run_approved"),
            "run_inference": execution.get("run_inference"),
            "create_submission_file": execution.get("create_submission_file"),
            "generate_submission_file_approved": nested(
                config, "submission_plan.generate_submission_file_approved"
            ),
        }
        missing = [name for name, enabled in required.items() if not bool(enabled)]
        if missing:
            raise RuntimeError(
                "Kaggle inference/submission-file generation is not approved: "
                + ", ".join(missing)
            )

    scientific_contract = {
        "candidate": ACTIVE_VARIANT,
        "particles": PARTICLES,
        "seeds": SEEDS,
        "seed_indices": [0, 127],
        "seed_namespace": SEED_NAMESPACE,
        "seed_formula": "stable_seed('likpf','test',well)+seed_index",
        "temperature": TEMPERATURE,
        "gr_scale_multiplier": GR_SCALE_MULTIPLIER,
        "alternative_aggregations_generated": 0,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "parent_control_reruns": 0,
        "source_pins": source_pins,
    }
    scientific_contract["sha256"] = canonical_json_sha256(scientific_contract)
    return scientific_contract


# %% [markdown]
# ## 3. Runtime, path, SHA, and identity helpers

# %%
def locate_package_dir() -> Path:
    cwd = Path.cwd()
    if (cwd / "config.yaml").is_file():
        return cwd
    candidate = cwd / "experiments" / EXPERIMENT_NAME
    if (candidate / "config.yaml").is_file():
        return candidate
    raise FileNotFoundError(f"cannot locate {EXPERIMENT_NAME}/config.yaml from {cwd}")


PACKAGE_DIR = locate_package_dir()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")


def load_config() -> dict[str, Any]:
    value = yaml.safe_load((PACKAGE_DIR / "config.yaml").read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError("config.yaml must contain a mapping")
    return value


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def sha256_gzip_content(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _normalize_frame_for_hash(frame: pd.DataFrame) -> pd.DataFrame:
    string_columns = [
        column
        for column, dtype in frame.dtypes.items()
        if isinstance(dtype, pd.StringDtype)
    ]
    if not string_columns:
        return frame
    normalized = frame.copy()
    for column in string_columns:
        normalized[column] = normalized[column].astype(object)
    return normalized


def frame_content_sha256(frame: pd.DataFrame) -> str:
    normalized = _normalize_frame_for_hash(frame)
    digest = hashlib.sha256()
    digest.update("|".join(normalized.columns).encode())
    digest.update("|".join(str(dtype) for dtype in normalized.dtypes).encode())
    row_hashes = pd.util.hash_pandas_object(
        normalized, index=False, categorize=True
    )
    digest.update(
        row_hashes.to_numpy(dtype="uint64").astype("<u8", copy=False).tobytes()
    )
    return digest.hexdigest()


def schema_sha256(frame: pd.DataFrame) -> str:
    normalized = _normalize_frame_for_hash(frame)
    schema = [(column, str(dtype)) for column, dtype in normalized.dtypes.items()]
    return canonical_json_sha256(schema)


def source_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


@dataclass(frozen=True)
class RuntimePaths:
    data_root: Path
    test_dir: Path
    sample_submission: Path
    output_root: Path
    artifacts_dir: Path
    submission: Path
    metrics: Path


def resolve_data_root() -> Path:
    local = PACKAGE_DIR.parents[1] / "data" / "raw"
    if (local / "sample_submission.csv").is_file() and (local / "test").is_dir():
        return local
    if KAGGLE_INPUT_ROOT.is_dir():
        candidates = sorted(
            {
                path.parent
                for path in KAGGLE_INPUT_ROOT.rglob("sample_submission.csv")
                if (path.parent / "test").is_dir()
            }
        )
        if len(candidates) == 1:
            return candidates[0]
        competition = [path for path in candidates if "rogii" in str(path).lower()]
        if len(competition) == 1:
            return competition[0]
        raise FileNotFoundError(
            f"expected one competition data root, got {candidates}"
        )
    raise FileNotFoundError("competition test/sample_submission root not found")


def build_runtime_paths() -> RuntimePaths:
    data_root = resolve_data_root()
    kaggle_runtime = KAGGLE_WORKING_ROOT.is_dir()
    output_root = KAGGLE_WORKING_ROOT if kaggle_runtime else PACKAGE_DIR
    artifacts_dir = output_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return RuntimePaths(
        data_root=data_root,
        test_dir=data_root / "test",
        sample_submission=data_root / "sample_submission.csv",
        output_root=output_root,
        artifacts_dir=artifacts_dir,
        submission=output_root / "submission.csv",
        metrics=output_root / "metrics.json",
    )


def parse_identity(frame: pd.DataFrame) -> pd.DataFrame:
    ids = frame["id"].astype(str)
    split = ids.str.rsplit("_", n=1, expand=True)
    if split.shape[1] != 2:
        raise ValueError("id must use <well>_<row_idx>")
    rows = pd.to_numeric(split[1], errors="raise")
    if (rows < 0).any() or not np.equal(rows, np.floor(rows)).all():
        raise ValueError("submission row indices must be non-negative integers")
    return pd.DataFrame(
        {
            "id": ids.astype(object),
            "well": split[0].astype(str).astype(object),
            "well_row_idx": rows.astype(np.int32),
        }
    )


# %% [markdown]
# ## 4. Sample submission and public-reference helpers

# %%
def validate_sample_submission(sample: pd.DataFrame) -> dict[str, Any]:
    if list(sample.columns) != ["id", "tvt"]:
        raise ValueError(
            f"sample submission columns must be ['id', 'tvt'], got {list(sample.columns)}"
        )
    identity = parse_identity(sample)
    if identity["id"].duplicated().any():
        raise ValueError("sample submission contains duplicate ids")
    if identity[["well", "well_row_idx"]].duplicated().any():
        raise ValueError("sample submission contains duplicate well/row identity")
    if identity.empty or identity["well"].nunique() < 1:
        raise ValueError("sample submission must contain at least one nonempty well")
    return {
        "rows": int(len(identity)),
        "wells": int(identity["well"].nunique()),
        "unique_ids": int(identity["id"].nunique()),
        "schema_sha256": schema_sha256(sample),
        "id_content_sha256": frame_content_sha256(identity[["id"]]),
    }


def validate_raw_test_files(test_dir: Path, wells: list[str]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for well in wells:
        horizontal = test_dir / f"{well}__horizontal_well.csv"
        typewell = test_dir / f"{well}__typewell.csv"
        if not horizontal.is_file() or not typewell.is_file():
            raise FileNotFoundError(f"raw-test file pair missing for well {well}")
        records.append(
            {
                "well": well,
                "horizontal": source_record(horizontal),
                "typewell": source_record(typewell),
            }
        )
    return {
        "well_count": len(wells),
        "records": records,
        "manifest_sha256": canonical_json_sha256(records),
    }


def resolve_public_reference(config: Mapping[str, Any]) -> Path:
    reference = nested(config, "data.public_reference")
    local_path = PACKAGE_DIR.parents[1] / str(reference["local_path"])
    if local_path.is_file():
        return local_path
    filename = str(reference["filename"])
    token = str(reference["source_path_token"])
    matches = [
        path
        for path in sorted(KAGGLE_INPUT_ROOT.rglob(filename))
        if token in str(path)
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one public parity reference {filename} under {token}, "
            f"got {matches}"
        )
    return matches[0]


# %% [markdown]
# ## 5. Frozen exp073 likelihood-PF kernel
#
# `stable_seed`、`_interp1`、`_grid`、`_pf_lik_allseeds`は、exp413 v4が
# kernel sourceとして使ったexp073 source SHA `4af212...`からそのまま抽出した。
# Numba kernel内でwellごとのseed baseとseed indexを明示的にseedし、worker順に
# 依存しない。集約は次章でtemperature 5だけをmaterializeする。

# %%
class CFG:
    DATA = Path(os.environ.get("ROGII_DATA", "data/raw"))
    OUT = Path(os.environ.get("ROGII_OUT", "."))
    seed = BASE_SEED
    n_jobs = min(8, multiprocessing.cpu_count())
    PF_SEEDS = SEEDS
    PF_PARTICLES = PARTICLES


def stable_seed(*parts: object, modulo: int = 2_147_483_647) -> int:
    key = "::".join(str(part) for part in parts)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % modulo + 1


def configure_public_runtime(
    *,
    data_dir: str | Path,
    output_dir: str | Path,
    n_jobs: int,
    pf_seeds: int,
    pf_particles: int,
) -> None:
    if int(pf_seeds) != SEEDS or int(pf_particles) != PARTICLES:
        raise ValueError("PF seeds/particles differ from the frozen exp452 contract")
    CFG.DATA = Path(data_dir)
    CFG.OUT = Path(output_dir)
    CFG.n_jobs = int(n_jobs)
    CFG.PF_SEEDS = int(pf_seeds)
    CFG.PF_PARTICLES = int(pf_particles)


def warm_up_likpf_kernel() -> None:
    """Compile the frozen Numba kernel before well-level thread parallelism."""
    md = np.linspace(1.0, 50.0, 20)
    z = np.zeros(20)
    gr = np.full(20, 50.0)
    grid = np.linspace(45.0, 55.0, 100)
    _pf_lik_allseeds(
        md,
        z,
        gr,
        grid,
        45.0,
        0.1,
        20.0,
        50.0,
        0.0,
        64,
        4,
        0,
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        4.5,
    )


def load_well(wid: str, split: str = "test") -> tuple[pd.DataFrame, pd.DataFrame]:
    base = CFG.DATA / split
    hw = pd.read_csv(base / f"{wid}__horizontal_well.csv")
    tw = pd.read_csv(base / f"{wid}__typewell.csv").sort_values("TVT")
    return hw, tw


@njit(cache=True)
def _interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0: return grid[0]
    n = len(grid) - 1
    if i >= n: return grid[n]
    t = (v - vmin) / step - i
    return grid[i]*(1.-t) + grid[i+1]*t


def _grid(tw_tvt, tw_gr, step=0.2):
    tmin = float(tw_tvt.min()); tmax = float(tw_tvt.max())
    tvt_g = np.arange(tmin, tmax+step, step)
    return np.interp(tvt_g, tw_tvt, tw_gr).astype(np.float64), float(tmin), float(step)


@njit(cache=True, nogil=True)
def _pf_lik_allseeds(md_v, z_v, gr_v, gg, vmin, step, gs, ls, ir, N, n_seeds, seed_base,
                     MOM, VN, PN, RP, RR, RESAMP, init_spr):
    n = len(md_v); preds = np.empty((n_seeds, n)); liks = np.empty(n_seeds); tmax = vmin + len(gg)*step
    for s in range(n_seeds):
        np.random.seed(seed_base + s)
        pos = np.empty(N); rate = np.empty(N); w = np.ones(N)/N
        for j in range(N):
            pos[j] = ls + init_spr*np.random.randn(); rate[j] = ir + 0.01*np.random.randn()
        log_lik = 0.0; prev_md = md_v[0] - 1.0
        for i in range(n):
            dm = md_v[i] - prev_md
            if dm < 1.0: dm = 1.0
            for j in range(N):
                rate[j] = MOM*rate[j] + VN*np.random.randn(); pos[j] += rate[j]*dm + PN*np.random.randn()
                tvt_j = pos[j] - z_v[i]
                if tvt_j < vmin-100.: tvt_j = vmin-100.
                if tvt_j > tmax+100.: tvt_j = tmax+100.
                pos[j] = tvt_j + z_v[i]
            avg_lk = 0.0
            for j in range(N):
                eg = _interp1(gg, pos[j]-z_v[i], vmin, step); d = (gr_v[i]-eg)/gs; dd = d*d
                if dd > 600.: dd = 600.
                lk = np.exp(-0.5*dd)
                if lk < 1e-300: lk = 1e-300
                avg_lk += w[j]*lk; w[j] = w[j]*lk
            if avg_lk < 1e-300: avg_lk = 1e-300
            log_lik += np.log(avg_lk)
            ws = 0.0
            for j in range(N): ws += w[j]
            if ws > 0.0:
                for j in range(N): w[j] /= ws
            else:
                for j in range(N): w[j] = 1./N
            neff = 0.0
            for j in range(N): neff += w[j]*w[j]
            neff = 1.0/neff
            if neff < RESAMP*N:
                cum = np.empty(N); c = 0.0
                for j in range(N): c += w[j]; cum[j] = c
                u0 = np.random.uniform(0., 1./N); newpos = np.empty(N); newrate = np.empty(N); ci = 0
                for j in range(N):
                    u = u0 + j/N
                    while ci < N-1 and cum[ci] < u: ci += 1
                    newpos[j] = pos[ci] + RP*np.random.randn(); newrate[j] = rate[ci] + RR*np.random.randn()
                for j in range(N): pos[j] = newpos[j]; rate[j] = newrate[j]; w[j] = 1./N
            est = 0.0
            for j in range(N): est += w[j]*(pos[j]-z_v[i])
            preds[s, i] = est; prev_md = md_v[i]
        liks[s] = log_lik
    return preds, liks


# %% [markdown]
# ## 6. Scale-5-only raw-test generation helpers

# %%
def lik_pf_scale5(
    hw: pd.DataFrame,
    tw: pd.DataFrame,
    *,
    n_particles: int = PARTICLES,
    n_seeds: int = SEEDS,
    temperature: float = TEMPERATURE,
    init_spr: float = 4.5,
    seed_base: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if n_particles != PARTICLES or n_seeds != SEEDS:
        raise ValueError("scale-5 generator must use 500 particles and 128 seeds")
    if temperature != TEMPERATURE:
        raise ValueError("scale-5 generator temperature must remain 5")
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s.TVT.values.astype(float)
    tw_gr = tw_s.GR.fillna(tw_s.GR.mean()).values.astype(float)
    kn = hw[hw.TVT_input.notna()]
    ev = hw[hw.TVT_input.isna()]
    if len(kn) == 0:
        raise ValueError("horizontal well has no known TVT_input prefix")
    if len(ev) == 0:
        raise ValueError("horizontal well has no unknown suffix rows")
    last = kn.iloc[-1]
    ls = float(last.TVT_input) + float(last.Z)
    tw_at_k = np.interp(kn.TVT_input.values, tw_tvt, tw_gr)
    gs = float(
        np.clip(np.nanstd(kn.GR.fillna(0).values - tw_at_k), 10.0, 60.0)
    )
    tail = kn.tail(30)
    dt = np.diff(tail.TVT_input.values)
    dz = np.diff(tail.Z.values)
    dm = np.diff(tail.MD.values)
    valid_step = dm > 0
    ir = (
        float(np.median((dt + dz)[valid_step] / dm[valid_step]))
        if valid_step.sum() >= 3
        else 0.0
    )
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    gr_v = (
        hw.GR.interpolate(limit_direction="both")
        .fillna(tw_gr.mean())
        .values.astype(float)[ev.index]
    )
    preds, log_likelihood = _pf_lik_allseeds(
        ev.MD.values.astype(float),
        ev.Z.values.astype(float),
        gr_v,
        gg,
        gmin,
        gst,
        gs,
        ls,
        ir,
        n_particles,
        n_seeds,
        seed_base,
        0.998,
        0.002,
        0.005,
        0.1,
        0.001,
        0.5,
        init_spr,
    )
    centered = log_likelihood - log_likelihood.max()
    weights = np.exp(centered / temperature)
    weights /= weights.sum()
    scale5 = (weights[:, None] * preds).sum(axis=0).astype(np.float32)
    if not np.isfinite(scale5).all():
        raise ValueError("scale-5 likelihood-PF produced non-finite values")
    audit = {
        "known_rows": int(len(kn)),
        "predicted_rows": int(len(ev)),
        "gr_sigma": gs,
        "initial_rate": ir,
        "log_likelihood_max": float(log_likelihood.max()),
        "log_likelihood_std": float(log_likelihood.std()),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_sum": float(weights.sum()),
        "materialized_aggregations": [ACTIVE_VARIANT],
    }
    return scale5, ev.index.to_numpy(np.int32), audit


def _scale5_rows(well: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    hw, tw = load_well(well, SPLIT_NAME)
    seed_base = stable_seed(FEATURE_FAMILY, SPLIT_NAME, well)
    values, row_indices, audit = lik_pf_scale5(
        hw,
        tw,
        seed_base=seed_base,
    )
    frame = pd.DataFrame(
        {
            "id": [f"{well}_{int(index)}" for index in row_indices],
            "well": [well] * len(row_indices),
            "well_row_idx": row_indices,
            "candidate_tvt": values,
        }
    )
    frame["id"] = frame["id"].astype(object)
    frame["well"] = frame["well"].astype(object)
    frame["well_row_idx"] = frame["well_row_idx"].astype(np.int32)
    frame["candidate_tvt"] = frame["candidate_tvt"].astype(np.float32)
    audit.update(
        {
            "well": well,
            "seed_base": int(seed_base),
            "seed_first": int(seed_base),
            "seed_last": int(seed_base + SEEDS - 1),
            "prediction_content_sha256": frame_content_sha256(frame),
        }
    )
    return frame, audit


def build_scale5_surface(
    wells: list[str],
    *,
    n_jobs: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if not wells or len(set(wells)) != len(wells):
        raise ValueError("well list must be nonempty and unique")
    outputs = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_scale5_rows)(well) for well in wells
    )
    frames = [frame for frame, _ in outputs]
    audits = [audit for _, audit in outputs]
    result = pd.concat(frames, ignore_index=True)
    if result.empty or result["id"].duplicated().any():
        raise ValueError("scale-5 surface is empty or contains duplicate ids")
    if set(result["well"].astype(str)) != set(wells):
        raise ValueError("scale-5 surface well set differs from requested wells")
    if len({int(item["seed_base"]) for item in audits}) != len(audits):
        raise ValueError("stable per-well seed bases are not unique")
    return result, audits


# %% [markdown]
# ## 7. Parity, submission, and manifest helpers

# %%
def candidate_hash_frame(candidate: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": candidate["id"].astype(str).astype(object),
            "well": candidate["well"].astype(str).astype(object),
            "candidate_tvt": candidate["candidate_tvt"].to_numpy(np.float32),
        }
    )


def load_public_reference(path: Path) -> pd.DataFrame:
    reference = pd.read_csv(
        path,
        usecols=["id", "well", "likpf_scale_5"],
        dtype={"id": str, "well": str},
    )
    reference = reference.rename(columns={"likpf_scale_5": "candidate_tvt"})
    reference = reference[["id", "well", "candidate_tvt"]]
    reference["id"] = reference["id"].astype(str).astype(object)
    reference["well"] = reference["well"].astype(str).astype(object)
    reference["candidate_tvt"] = reference["candidate_tvt"].astype(np.float32)
    if reference["id"].duplicated().any():
        raise ValueError("public reference contains duplicate ids")
    return reference


def validate_public_reference_parity(
    candidate: pd.DataFrame,
    reference_path: Path,
) -> dict[str, Any]:
    file_sha = sha256_file(reference_path)
    content_sha = sha256_gzip_content(reference_path)
    if file_sha != EXPECTED_PUBLIC_REFERENCE_GZIP_SHA256:
        raise ValueError("public reference gzip SHA mismatch")
    if content_sha != EXPECTED_PUBLIC_REFERENCE_CONTENT_SHA256:
        raise ValueError("public reference decompressed SHA mismatch")
    reference = load_public_reference(reference_path)
    reference_logical_sha = frame_content_sha256(reference)
    if reference_logical_sha != EXPECTED_PUBLIC_CANDIDATE_CONTENT_SHA256:
        raise ValueError("public reference logical candidate SHA mismatch")

    candidate_ids = set(candidate["id"].astype(str))
    reference_ids = set(reference["id"].astype(str))
    audit: dict[str, Any] = {
        "reference_path": str(reference_path),
        "reference_file_sha256": file_sha,
        "reference_decompressed_sha256": content_sha,
        "reference_candidate_content_sha256": reference_logical_sha,
        "reference_rows": int(len(reference)),
        "reference_wells": int(reference["well"].nunique()),
    }
    if candidate_ids != reference_ids:
        audit.update(
            {
                "status": "skipped_hidden_id_set_differs_from_public_reference",
                "candidate_rows": int(len(candidate)),
                "candidate_wells": int(candidate["well"].nunique()),
                "float32_max_abs_ft": None,
            }
        )
        return audit

    left = candidate_hash_frame(candidate).set_index("id")
    right = reference.set_index("id")
    right = right.loc[left.index]
    if not left["well"].equals(right["well"]):
        raise ValueError("public reference well identity differs")
    delta = np.abs(
        left["candidate_tvt"].to_numpy(np.float32)
        - right["candidate_tvt"].to_numpy(np.float32)
    )
    max_abs = float(delta.max(initial=0.0))
    candidate_logical_sha = frame_content_sha256(candidate_hash_frame(candidate))
    if max_abs != 0.0:
        raise ValueError(f"exp413 v4 float32 parity failed: max_abs={max_abs}")
    if candidate_logical_sha != EXPECTED_PUBLIC_CANDIDATE_CONTENT_SHA256:
        raise ValueError("generated public candidate logical SHA mismatch")
    audit.update(
        {
            "status": "passed_public_float32_exact_parity",
            "candidate_content_sha256": candidate_logical_sha,
            "candidate_rows": int(len(candidate)),
            "candidate_wells": int(candidate["well"].nunique()),
            "float32_max_abs_ft": max_abs,
        }
    )
    return audit


def build_submission(sample: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    validate_sample_submission(sample)
    if candidate["id"].duplicated().any():
        raise ValueError("candidate surface contains duplicate ids")
    sample_ids = sample["id"].astype(str).reset_index(drop=True)
    candidate_ids = candidate["id"].astype(str)
    if len(sample_ids) != len(candidate_ids) or set(sample_ids) != set(candidate_ids):
        raise ValueError("candidate IDs do not match sample submission")
    value_by_id = pd.Series(
        candidate["candidate_tvt"].to_numpy(np.float32),
        index=candidate_ids,
    )
    values = value_by_id.reindex(sample_ids)
    if values.isna().any() or not np.isfinite(values.to_numpy(float)).all():
        raise ValueError("candidate mapping is incomplete or non-finite; fallback is forbidden")
    output = pd.DataFrame(
        {
            "id": sample_ids.astype(object),
            "tvt": values.to_numpy(np.float32),
        }
    )
    if list(output.columns) != ["id", "tvt"] or output["id"].duplicated().any():
        raise ValueError("submission schema/uniqueness contract failed")
    return output


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n")


# %% [markdown]
# ## 8. Setup and frozen configuration preview

# %%
IMPORT_ONLY = os.environ.get(IMPORT_ONLY_ENV) == "1"

if not IMPORT_ONLY:
    started_at = time.time()
    config = load_config()
    scientific_contract = validate_frozen_contract(
        config,
        require_execution_approval=True,
    )
    paths = build_runtime_paths()
    sample = pd.read_csv(paths.sample_submission, dtype={"id": str})
    sample["id"] = sample["id"].astype(str).astype(object)
    sample_audit = validate_sample_submission(sample)
    identity = parse_identity(sample)
    test_wells = sorted(identity["well"].unique().tolist())
    input_manifest = validate_raw_test_files(paths.test_dir, test_wells)
    reference_path = resolve_public_reference(config)
    configure_public_runtime(
        data_dir=paths.data_root,
        output_dir=paths.artifacts_dir,
        n_jobs=int(nested(config, "runtime.num_workers")),
        pf_seeds=SEEDS,
        pf_particles=PARTICLES,
    )
    warm_up_likpf_kernel()
    display(
        {
            "experiment": EXPERIMENT_NAME,
            "route": nested(config, "experiment.route"),
            "status": nested(config, "experiment.status"),
            "parent": nested(config, "lineage.parent"),
            "candidate": ACTIVE_VARIANT,
            "sample": sample_audit,
            "test_wells_dynamic": len(test_wells),
            "particles": PARTICLES,
            "seeds": SEEDS,
            "temperature": TEMPERATURE,
            "n_jobs": CFG.n_jobs,
            "train_runs": 0,
            "model_configs": 0,
            "trained_folds": 0,
            "boosters": 0,
            "alternative_aggregations": 0,
            "competition_submission": "not_approved_not_started",
        }
    )
    display(identity.head(20))


# %% [markdown]
# ## 9. Generate the one scale-5 candidate

# %%
if not IMPORT_ONLY:
    candidate, per_well_audits = build_scale5_surface(
        test_wells,
        n_jobs=CFG.n_jobs,
    )
    if len(candidate) != len(sample):
        raise ValueError("scale-5 candidate row count differs from sample submission")
    if set(candidate["id"].astype(str)) != set(sample["id"].astype(str)):
        raise ValueError("scale-5 candidate ID set differs from sample submission")
    candidate_content_sha = frame_content_sha256(candidate_hash_frame(candidate))
    generation_audit = {
        "candidate": ACTIVE_VARIANT,
        "rows": int(len(candidate)),
        "wells": int(candidate["well"].nunique()),
        "particles_per_seed": PARTICLES,
        "seeds_per_well": SEEDS,
        "well_seed_runs": int(len(test_wells) * SEEDS),
        "particle_trajectories": int(len(test_wells) * SEEDS * PARTICLES),
        "temperature": TEMPERATURE,
        "seed_namespace": SEED_NAMESPACE,
        "thread_schedule_independent": True,
        "fallback_rows": 0,
        "fallback_wells": 0,
        "alternative_aggregations_generated": 0,
        "candidate_content_sha256": candidate_content_sha,
        "per_well": per_well_audits,
    }
    display(generation_audit)
    display(candidate.head(20))


# %% [markdown]
# ## 10. Public-reference parity and submission generation

# %%
if not IMPORT_ONLY:
    parity_audit = validate_public_reference_parity(candidate, reference_path)
    submission = build_submission(sample, candidate)
    submission.to_csv(paths.submission, index=False)
    written = pd.read_csv(paths.submission, dtype={"id": str})
    if not written["id"].astype(str).equals(sample["id"].astype(str)):
        raise ValueError("written submission order differs from sample submission")
    if written["tvt"].isna().any() or not np.isfinite(written["tvt"]).all():
        raise ValueError("written submission contains missing or non-finite values")

    prediction_path = paths.artifacts_dir / "scale5_likpf_prediction.csv.gz"
    candidate.to_csv(
        prediction_path,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    submission_sha = sha256_file(paths.submission)
    prediction_file_sha = sha256_file(prediction_path)
    prediction_decompressed_sha = sha256_gzip_content(prediction_path)
    prediction_stats = {
        "rows": int(len(submission)),
        "wells": int(candidate["well"].nunique()),
        "fallback_rows": 0,
        "fallback_wells": 0,
        "duplicate_ids": int(submission["id"].duplicated().sum()),
        "missing_values": int(submission["tvt"].isna().sum()),
        "nonfinite_values": int((~np.isfinite(submission["tvt"])).sum()),
        "min": float(submission["tvt"].min()),
        "max": float(submission["tvt"].max()),
        "mean": float(submission["tvt"].mean()),
        "std": float(submission["tvt"].std()),
    }
    display(parity_audit)
    display(prediction_stats)
    display(submission.head(20))


# %% [markdown]
# ## 11. Metrics, SHA, and generated artifacts

# %%
if not IMPORT_ONLY:
    input_manifest_path = paths.artifacts_dir / "input_manifest.json"
    generation_manifest_path = paths.artifacts_dir / "generation_manifest.json"
    write_json(input_manifest_path, input_manifest)

    generation_manifest = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": "generated_pending_submit_check_and_competition_submission_approval",
        "scientific_contract": scientific_contract,
        "config_sha256": sha256_file(PACKAGE_DIR / "config.yaml"),
        "notebook_source_sha256": sha256_file(
            PACKAGE_DIR
            / f"{EXPERIMENT_NAME}_compact_selfcontained_inference.py"
        ),
        "sample_audit": sample_audit,
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "generation": generation_audit,
        "public_reference_parity": parity_audit,
        "prediction_file": str(prediction_path),
        "prediction_file_sha256": prediction_file_sha,
        "prediction_decompressed_sha256": prediction_decompressed_sha,
        "prediction_content_sha256": candidate_content_sha,
        "submission_file": str(paths.submission),
        "submission_sha256": submission_sha,
        "prediction_stats": prediction_stats,
        "model_manifest": {
            "model_count": 0,
            "model_sha256": None,
            "trained_folds": 0,
            "boosters": 0,
        },
        "submit_check": "pending",
        "competition_submission": "not_approved_not_started",
        "public_lb": None,
        "deterministic_anchor": False,
        "runtime_seconds": round(time.time() - started_at, 3),
    }
    write_json(generation_manifest_path, generation_manifest)

    metrics = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": "inference_generated_pending_submit_check",
        "candidate_id": ACTIVE_VARIANT,
        "cv": float(nested(config, "validation.train_side_evidence.candidate_rmse")),
        "rows": int(len(submission)),
        "wells": int(candidate["well"].nunique()),
        "particles": PARTICLES,
        "seeds": SEEDS,
        "temperature": TEMPERATURE,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "parent_control_reruns": 0,
        "fallback_rows": 0,
        "fallback_wells": 0,
        "sample_audit": sample_audit,
        "input_manifest_sha256": sha256_file(input_manifest_path),
        "scientific_contract_sha256": scientific_contract["sha256"],
        "generation_manifest_sha256": sha256_file(generation_manifest_path),
        "public_reference_parity": parity_audit,
        "prediction_file_sha256": prediction_file_sha,
        "prediction_decompressed_sha256": prediction_decompressed_sha,
        "prediction_content_sha256": candidate_content_sha,
        "submission_sha256": submission_sha,
        "prediction_stats": prediction_stats,
        "runtime_seconds": round(time.time() - started_at, 3),
        "submit_check": "pending",
        "competition_submission": "not_approved_not_started",
        "public_lb": None,
        "private_lb": None,
        "deterministic_anchor": False,
    }
    write_json(paths.metrics, metrics)
    write_json(paths.artifacts_dir / "inference_metrics.json", metrics)

    print("Generated artifacts:")
    for generated_path in (
        input_manifest_path,
        generation_manifest_path,
        prediction_path,
        paths.metrics,
        paths.submission,
    ):
        print(f"- {generated_path} ({generated_path.stat().st_size} bytes)")
