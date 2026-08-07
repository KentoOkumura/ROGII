# %% [markdown]
# # exp305 tempered raw-smoothed exact-HMM emission
#
# Train-side only, fixed-contract decoder audit. The sole scientific variant
# mixes the exp304-selected SWT and raw Gaussian log emissions with fixed
# weights 0.15/0.85, then runs the unchanged exp209 exact-HMM posterior mean.

# %% [markdown]
# ## Contents
# 1. Imports and execution guard
# 2. Runtime, configuration, path, and SHA helpers
# 3. Frozen scientific contract and input preflight
# 4. Exp304 selected-series streaming and raw-input parity
# 5. Tempered emission and exact exp209 forward-backward kernel
# 6. Target-free well decoding and prediction freeze
# 7. Late truth/control attachment and paired metrics
# 8. Promotion gate and generated artifacts
# 9. Setup, configuration, and contract preview
# 10. Run the Kaggle CPU audit

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd
import yaml

try:
    from numba import get_num_threads, njit, prange, set_num_threads

    NUMBA_AVAILABLE = True
except ModuleNotFoundError:
    NUMBA_AVAILABLE = False

    def prange(*args: Any) -> range:
        return range(*args)

    def set_num_threads(_: int) -> None:
        return None

    def get_num_threads() -> int | None:
        return None

    def njit(*args: Any, **_: Any) -> Any:
        if args and callable(args[0]):
            return args[0]

        def decorator(function: Any) -> Any:
            return function

        return decorator


EXPERIMENT_NAME = "exp305_tempered_raw_smoothed_exact_hmm_emission"
OUTPUT_PREFIX = EXPERIMENT_NAME
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
PREDICTION_COLUMNS = [
    "id",
    "well_id",
    "row_idx",
    "tempered_hmm_tvt",
    "tempered_hmm_std",
]


def in_notebook_runtime() -> bool:
    try:
        return get_ipython() is not None  # type: ignore[name-defined]
    except NameError:
        return False


EXECUTE_NOTEBOOK = os.environ.get("EXP305_IMPORT_ONLY", "0") != "1" and in_notebook_runtime()


# %% [markdown]
# ## 2. Runtime, configuration, path, and SHA helpers


# %%
def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        item = float(value)
        return item if math.isfinite(item) else None
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    try:
        if pd.isna(value) and not isinstance(value, str):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def get_nested(config: dict[str, Any], dotted_key: str) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def project_root() -> Path:
    start = Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "project.yml").exists():
            return candidate
    return start


def load_experiment_config() -> dict[str, Any]:
    root = project_root()
    candidates = (
        Path.cwd() / "config.yaml",
        root / "experiments" / EXPERIMENT_NAME / "config.yaml",
    )
    for path in candidates:
        config = read_yaml(path)
        if get_nested(config, "experiment.name") == EXPERIMENT_NAME:
            return config
    raise FileNotFoundError(f"exp305 config not found in {[str(path) for path in candidates]}")


def artifact_dir() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        path = KAGGLE_WORKING_ROOT / "artifacts"
    else:
        path = project_root() / "experiments" / EXPERIMENT_NAME / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def metrics_output_path() -> Path:
    if KAGGLE_WORKING_ROOT.exists():
        return KAGGLE_WORKING_ROOT / "metrics.json"
    return project_root() / "experiments" / EXPERIMENT_NAME / "metrics.json"


def train_data_dir(config: dict[str, Any]) -> Path:
    if KAGGLE_INPUT_ROOT.exists():
        fixed = (
            KAGGLE_INPUT_ROOT / "rogii-wellbore-geology-prediction" / "train",
            KAGGLE_INPUT_ROOT / "competitions" / "rogii-wellbore-geology-prediction" / "train",
        )
        for candidate in fixed:
            if next(candidate.glob("*__horizontal_well.csv"), None) is not None:
                return candidate
        for candidate in sorted(KAGGLE_INPUT_ROOT.glob("**/train")):
            if next(candidate.glob("*__horizontal_well.csv"), None) is not None:
                return candidate
    return project_root() / str(get_nested(config, "data.train_dir") or "data/raw/train")


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_gzip_csv(path: str | Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    newline_count = 0
    last_byte = b""
    with gzip.open(path, "rb") as file_pointer:
        for chunk in iter(lambda: file_pointer.read(1024 * 1024), b""):
            digest.update(chunk)
            newline_count += chunk.count(b"\n")
            if chunk:
                last_byte = chunk[-1:]
    line_count = newline_count + int(bool(last_byte) and last_byte != b"\n")
    columns = [str(column) for column in pd.read_csv(path, nrows=0).columns]
    return {
        "path": str(path),
        "bytes": Path(path).stat().st_size,
        "raw_sha256": sha256_path(path),
        "decompressed_sha256": digest.hexdigest(),
        "data_rows": max(0, line_count - 1),
        "columns": columns,
    }


def require_csv_columns(report: dict[str, Any], required: Iterable[str], label: str) -> None:
    actual = {str(column) for column in report.get("columns", [])}
    missing = sorted({str(column) for column in required}.difference(actual))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def mapping_sha256(value: dict[str, Any]) -> str:
    payload = json.dumps(to_jsonable(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def dataframe_content_sha(frame: pd.DataFrame, columns: list[str] | None = None) -> str:
    chosen = list(frame.columns) if columns is None else list(columns)
    digest = hashlib.sha256()
    for column in chosen:
        digest.update(column.encode())
        values = frame[column]
        if pd.api.types.is_numeric_dtype(values):
            array = np.ascontiguousarray(values.to_numpy())
            digest.update(str(array.dtype).encode())
            digest.update(array.tobytes())
        else:
            for value in values.astype(str):
                digest.update(value.encode())
                digest.update(b"\n")
    return digest.hexdigest()


def resolve_existing(filename: str, candidates: Iterable[str]) -> Path:
    root = project_root()
    checked: list[str] = []
    for raw in candidates:
        candidate = Path(str(raw))
        paths = (
            candidate if candidate.name == filename else candidate / filename,
            root / candidate if candidate.name == filename else root / candidate / filename,
            Path.cwd() / candidate
            if candidate.name == filename
            else Path.cwd() / candidate / filename,
        )
        for path in paths:
            checked.append(str(path))
            if path.exists() and path.is_file():
                return path
    if KAGGLE_INPUT_ROOT.exists():
        for path in sorted(KAGGLE_INPUT_ROOT.glob(f"**/{filename}")):
            if path.is_file():
                return path
    raise FileNotFoundError(f"could not resolve {filename}; checked={checked}")


def runtime_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "pyyaml": getattr(yaml, "__version__", "unknown"),
        "numba_available": NUMBA_AVAILABLE,
        "blas_threads": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }
    if NUMBA_AVAILABLE:
        import numba

        versions["numba"] = numba.__version__
        versions["numba_num_threads"] = get_num_threads()
    return versions


# %% [markdown]
# ## 3. Frozen scientific contract and input preflight


# %%
def fixed_hmm_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    hmm = get_nested(config, "model.hmm") or {}
    keys = (
        "step",
        "n_rates",
        "rate_span",
        "sig_r",
        "sig_p",
        "df",
        "emission",
        "lam",
        "sigma_mode",
        "start_sig",
        "r0_sig",
        "band_pad",
        "mom",
        "rate_center",
    )
    missing = [key for key in keys if key not in hmm]
    if missing:
        raise ValueError(f"model.hmm is missing fixed keys: {missing}")
    return {key: hmm[key] for key in keys}


def validate_scientific_contract(
    config: dict[str, Any], *, require_run_approval: bool = False
) -> None:
    expected_values = {
        "experiment.name": EXPERIMENT_NAME,
        "experiment.route": "pf_beam",
        "lineage.parent": "exp304_gr_denoiser_emission_separability_readout",
        "lineage.methodology_parent": "exp209_exp072_exp205_joint_exact_parity_fast_cache_generation",
        "model.active_variant_count": 1,
        "model.lightgbm_config_count": 0,
        "model.fold_training_count": 0,
        "model.booster_count": 0,
        "model.parent_control_retraining": False,
        "model.hmm.hmm_well_runs": 773,
        "model.hmm.emission": "gauss",
        "model.hmm.sigma_mode": "std",
        "model.hmm.rate_center": "zero",
        "runtime.outer_workers": 2,
        "runtime.numba_num_threads": 2,
        "runtime.kaggle.enable_gpu": False,
        "runtime.kaggle.enable_internet": False,
        "data.saved_controls.likpf_prediction_column": "likpf_mean_d",
        "data.saved_controls.likpf_prediction_representation": "delta_from_last_known_tvt",
        "inference.enabled": False,
        "inference.create_submission": False,
        "execution.create_submission": False,
    }
    for key, expected in expected_values.items():
        if get_nested(config, key) != expected:
            raise ValueError(f"exp305 fixed contract mismatch: {key} must be {expected!r}")
    expected_hmm = {
        "step": 0.35,
        "n_rates": 41,
        "rate_span": 0.10,
        "sig_r": 0.002,
        "sig_p": 0.02,
        "df": 4.0,
        "lam": 1.0,
        "start_sig": 0.75,
        "r0_sig": 0.01,
        "band_pad": 100.0,
        "mom": 0.998,
    }
    hmm = fixed_hmm_kwargs(config)
    for key, expected in expected_hmm.items():
        if float(hmm[key]) != expected:
            raise ValueError(f"exp305 fixes model.hmm.{key}={expected}")
    emission = get_nested(config, "model.emission") or {}
    if (
        float(emission.get("raw_weight", -1.0)) != 0.85
        or float(emission.get("swt_weight", -1.0)) != 0.15
        or [float(value) for value in emission.get("sigma_clip", [])] != [10.0, 60.0]
        or float(emission.get("log_likelihood_clip", 0.0)) != 600.0
        or bool(emission.get("recompute_sigma_from_swt", True))
    ):
        raise ValueError("exp305 fixes raw/SWT weights, raw sigma clip, and likelihood clip")
    variants = [
        str(value) for value in get_nested(config, "model.active_scientific_variants") or []
    ]
    if variants != ["tempered_raw_swt_beta015"]:
        raise ValueError("exp305 requires exactly the fixed beta-0.15 scientific variant")
    blend = get_nested(config, "model.blend") or {}
    if (
        not bool(blend.get("enabled"))
        or float(blend.get("tempered_hmm_weight", -1.0)) != 0.5
        or float(blend.get("saved_likpf_weight", -1.0)) != 0.5
        or bool(blend.get("weight_grid"))
    ):
        raise ValueError("exp305 fixes the saved-likPF 50/50 blend without a grid")
    if not bool(get_nested(config, "execution.implementation_approved")):
        raise ValueError("exp305 implementation approval must be recorded")
    if require_run_approval and not (
        bool(get_nested(config, "execution.kaggle_push_approved"))
        and bool(get_nested(config, "execution.run_train"))
    ):
        raise RuntimeError("exp305 Kaggle package/push/run is not approved")


def build_scientific_contract(config: dict[str, Any]) -> dict[str, Any]:
    contract = {
        "experiment": EXPERIMENT_NAME,
        "route": get_nested(config, "experiment.route"),
        "truth_attached": False,
        "scientific_variant": get_nested(config, "model.active_scientific_variants"),
        "emission": get_nested(config, "model.emission"),
        "hmm": get_nested(config, "model.hmm"),
        "blend": get_nested(config, "model.blend"),
        "saved_baselines": get_nested(config, "audit.saved_baselines"),
        "promotion": get_nested(config, "audit.promotion"),
        "execution_counts": {
            "active_variants": 1,
            "hmm_well_runs": 773,
            "models": 0,
            "lightgbm_configs": 0,
            "trained_folds": 0,
            "pf_runs": 0,
            "beam_runs": 0,
            "boosters": 0,
            "control_reruns": 0,
        },
        "truth_freeze_policy": get_nested(config, "validation.truth_attachment"),
        "forbidden": [
            "beta_sigma_clip_hmm_or_blend_grid",
            "raw_hmm_control_rerun",
            "likpf_rerun",
            "raw_test_prediction",
            "inference",
            "submission",
        ],
    }
    contract["scientific_contract_sha256"] = mapping_sha256(contract)
    return contract


def _spec_paths(spec: dict[str, Any]) -> list[str]:
    return [str(value) for value in spec.get("candidates", [])]


def preflight_exp304(config: dict[str, Any]) -> dict[str, Any]:
    spec = get_nested(config, "data.exp304_selected_series") or {}
    candidates = _spec_paths(spec)
    names = {
        "series": str(spec["filename"]),
        "manifest": str(spec["manifest_filename"]),
        "summary": str(spec["summary_filename"]),
        "scientific_contract": str(spec["scientific_contract_filename"]),
        "input_manifest": str(spec["input_manifest_filename"]),
    }
    paths = {name: resolve_existing(filename, candidates) for name, filename in names.items()}
    if bool(spec.get("require_nonzero_file_size")) and paths["series"].stat().st_size <= 0:
        raise ValueError("exp304 selected-series file is empty")
    series_report = inspect_gzip_csv(paths["series"])
    require_csv_columns(
        series_report,
        {
            "series_kind",
            "well_id",
            "position",
            "coordinate",
            "original_missing",
            "raw_gr",
            "swt_db4_l3_gr",
        },
        "exp304 selected-series",
    )
    if int(series_report["data_rows"]) != int(spec["expected_data_rows"]):
        raise ValueError("exp304 selected-series row count mismatch")
    if series_report["decompressed_sha256"] != str(spec["expected_content_sha256"]):
        raise ValueError("exp304 selected-series decompressed/content SHA mismatch")
    manifest = read_json(paths["manifest"])
    summary = read_json(paths["summary"])
    source_contract = read_json(paths["scientific_contract"])
    input_manifest = read_json(paths["input_manifest"])
    series_artifact = manifest.get("series_artifact") or {}
    if bool(spec.get("require_manifest_raw_decompressed_content_sha_match")):
        expected_triplet = {
            "raw_sha256": series_report["raw_sha256"],
            "decompressed_sha256": series_report["decompressed_sha256"],
            "content_sha256": series_report["decompressed_sha256"],
        }
        for key, expected in expected_triplet.items():
            if str(series_artifact.get(key)) != str(expected):
                raise ValueError(f"exp304 series manifest mismatch for {key}")
    if str(summary.get("selected_denoiser")) != str(spec["expected_selected_denoiser"]):
        raise ValueError("exp304 selected denoiser mismatch")
    silent_fallback_count = manifest.get("silent_fallback_count", -1)
    if int(silent_fallback_count) != int(
        spec["expected_silent_fallback_count"]
    ):
        raise ValueError("exp304 silent fallback count mismatch")
    if str(source_contract.get("scientific_contract_sha256")) != str(
        spec["expected_scientific_contract_sha256"]
    ):
        raise ValueError("exp304 scientific-contract content SHA mismatch")
    raw_identity = str(
        (input_manifest.get("raw_train") or {}).get("well_file_identity_content_sha256")
    )
    if raw_identity != str(spec["expected_raw_well_identity_sha256"]):
        raise ValueError("exp304 raw well-file identity mismatch")
    return {
        "paths": {key: str(value) for key, value in paths.items()},
        "series": series_report,
        "selected_denoiser": summary.get("selected_denoiser"),
        "silent_fallback_count": silent_fallback_count,
        "source_scientific_contract_sha256": source_contract.get("scientific_contract_sha256"),
        "raw_well_file_identity_content_sha256": raw_identity,
    }


def validate_raw_well_identity(config: dict[str, Any], raw_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for horizontal_path in sorted(raw_dir.glob("*__horizontal_well.csv")):
        well = horizontal_path.name.replace("__horizontal_well.csv", "")
        typewell_path = raw_dir / f"{well}__typewell.csv"
        if not typewell_path.exists():
            raise FileNotFoundError(typewell_path)
        rows.append(
            {
                "well_id": well,
                "horizontal_raw_sha256": sha256_path(horizontal_path),
                "typewell_raw_sha256": sha256_path(typewell_path),
            }
        )
    frame = pd.DataFrame(rows).sort_values("well_id", kind="mergesort").reset_index(drop=True)
    actual = dataframe_content_sha(
        frame,
        ["well_id", "horizontal_raw_sha256", "typewell_raw_sha256"],
    )
    expected = str(
        get_nested(config, "data.exp304_selected_series.expected_raw_well_identity_sha256")
    )
    if len(frame) != int(get_nested(config, "validation.expected_wells")) or actual != expected:
        raise ValueError("current raw train well-file identity does not match exp304")
    return {"path": str(raw_dir), "wells": len(frame), "content_sha256": actual}


def preflight_controls_and_assignments(config: dict[str, Any]) -> dict[str, Any]:
    control = get_nested(config, "data.saved_controls") or {}
    control_candidates = _spec_paths(control)
    hmm_path = resolve_existing(str(control["hmm_cache_filename"]), control_candidates)
    exp072_path = resolve_existing(str(control["exp072_cache_filename"]), control_candidates)
    hmm_report = inspect_gzip_csv(hmm_path)
    exp072_report = inspect_gzip_csv(exp072_path)
    require_csv_columns(
        hmm_report,
        {"id", "well", str(control["raw_hmm_prediction_column"])},
        "saved exp209 HMM cache",
    )
    require_csv_columns(
        exp072_report,
        {
            "id",
            "well",
            "last_known_tvt",
            "md_since",
            str(control["likpf_prediction_column"]),
        },
        "saved exp209 exp072 cache",
    )
    if hmm_report["decompressed_sha256"] != str(control["expected_hmm_decompressed_sha256"]):
        raise ValueError("saved exp209 HMM decompressed SHA mismatch")
    if exp072_report["decompressed_sha256"] != str(
        control["expected_exp209_v5_exp072_cache_decompressed_sha256"]
    ):
        raise ValueError("saved exp209-v5 exp072 cache decompressed SHA mismatch")
    fold = get_nested(config, "data.fold_assignment") or {}
    fold_path = resolve_existing(str(fold["filename"]), _spec_paths(fold))
    fold_report = inspect_gzip_csv(fold_path)
    require_csv_columns(
        fold_report,
        {
            *[str(value) for value in fold["safe_columns"]],
            *[str(value) for value in fold["truth_columns"]],
        },
        "exp226 OOF assignment",
    )
    if fold_report["decompressed_sha256"] != str(fold["expected_exp226_oof_decompressed_sha256"]):
        raise ValueError("exp226 OOF decompressed SHA mismatch")
    safe_columns = [str(value) for value in fold["safe_columns"]]
    safe = pd.read_csv(fold_path, usecols=safe_columns, dtype={"well_id": str})
    safe["well_id"] = safe["well_id"].astype(str)
    for column in ("row_idx", "suffix_offset", "fold"):
        safe[column] = pd.to_numeric(safe[column], errors="raise").astype(np.int64)
    safe["tvt_geop"] = pd.to_numeric(safe["tvt_geop"], errors="raise").astype(np.float64)
    safe = safe.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)
    if safe.duplicated(["well_id", "row_idx"]).any():
        raise ValueError("exp226 safe OOF identity is duplicated")
    if (
        len(safe) != int(get_nested(config, "validation.expected_rows"))
        or safe["well_id"].nunique() != int(get_nested(config, "validation.expected_wells"))
        or sorted(safe["fold"].unique().tolist())
        != [int(value) for value in get_nested(config, "validation.expected_folds")]
    ):
        raise ValueError("exp226 safe OOF row/well/fold coverage mismatch")
    hidden = get_nested(config, "data.hidden_like_assignment") or {}
    hidden_path = resolve_existing(str(hidden["filename"]), _spec_paths(hidden))
    hidden_sha = sha256_path(hidden_path)
    if hidden_sha != str(hidden["expected_sha256"]):
        raise ValueError("exp115 hidden-like assignment SHA mismatch")
    hidden_frame = pd.read_csv(hidden_path, dtype={"well_id": str})
    role_columns = [str(value) for value in (hidden.get("role_columns") or {}).values()]
    if not {"well_id", *role_columns}.issubset(hidden_frame.columns):
        raise ValueError("exp115 hidden-like assignment columns are incomplete")
    if hidden_frame["well_id"].duplicated().any():
        raise ValueError("exp115 hidden-like assignment has duplicate wells")
    return {
        "paths": {
            "saved_hmm": str(hmm_path),
            "saved_exp072": str(exp072_path),
            "fold_assignment": str(fold_path),
            "hidden_like_assignment": str(hidden_path),
        },
        "saved_hmm": hmm_report,
        "saved_exp072": exp072_report,
        "fold_assignment": {
            **fold_report,
            "rows": len(safe),
            "wells": int(safe["well_id"].nunique()),
            "folds": sorted(int(value) for value in safe["fold"].unique()),
            "well_ids": sorted(safe["well_id"].unique().tolist()),
        },
        "hidden_like_assignment": {
            "path": str(hidden_path),
            "bytes": hidden_path.stat().st_size,
            "raw_sha256": hidden_sha,
            "rows": len(hidden_frame),
            "wells": int(hidden_frame["well_id"].nunique()),
        },
    }


# %% [markdown]
# ## 4. Exp304 selected-series streaming and raw-input parity


# %%
def iter_exp304_well_series(
    path: Path, chunksize: int = 250_000
) -> Iterator[tuple[str, pd.DataFrame]]:
    columns = [
        "series_kind",
        "well_id",
        "position",
        "coordinate",
        "original_missing",
        "raw_gr",
        "swt_db4_l3_gr",
    ]
    pending = pd.DataFrame(columns=columns)
    seen: set[str] = set()
    for chunk in pd.read_csv(path, usecols=columns, dtype={"well_id": str}, chunksize=chunksize):
        chunk["well_id"] = chunk["well_id"].astype(str)
        work = pd.concat([pending, chunk], ignore_index=True) if len(pending) else chunk
        tail_well = str(work["well_id"].iloc[-1])
        complete = work.loc[work["well_id"] != tail_well]
        pending = work.loc[work["well_id"] == tail_well].copy()
        for well, group in complete.groupby("well_id", sort=False):
            well = str(well)
            if well in seen:
                raise ValueError(f"exp304 series is not contiguous for well={well}")
            seen.add(well)
            yield well, group.reset_index(drop=True)
    if len(pending):
        well = str(pending["well_id"].iloc[0])
        if well in seen:
            raise ValueError(f"exp304 series is not contiguous for final well={well}")
        yield well, pending.reset_index(drop=True)


def validate_and_split_series(well: str, series: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if set(series["series_kind"].astype(str).unique()) != {"horizontal", "typewell"}:
        raise ValueError(f"well={well} must contain horizontal and typewell exp304 series")
    parts: dict[str, pd.DataFrame] = {}
    for kind in ("horizontal", "typewell"):
        frame = series.loc[series["series_kind"].astype(str) == kind].copy()
        frame["position"] = pd.to_numeric(frame["position"], errors="raise").astype(np.int64)
        frame = frame.sort_values("position", kind="mergesort").reset_index(drop=True)
        if not np.array_equal(frame["position"].to_numpy(), np.arange(len(frame))):
            raise ValueError(f"well={well} {kind} exp304 positions are not contiguous")
        numeric = frame[["coordinate", "raw_gr", "swt_db4_l3_gr"]].apply(
            pd.to_numeric, errors="raise"
        )
        if not np.isfinite(numeric.to_numpy(np.float64)).all():
            raise ValueError(f"well={well} {kind} exp304 selected series is non-finite")
        frame[["coordinate", "raw_gr", "swt_db4_l3_gr"]] = numeric
        parts[kind] = frame
    return parts["horizontal"], parts["typewell"]


def load_raw_horizontal_without_truth(well: str, raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / f"{well}__horizontal_well.csv"
    frame = pd.read_csv(path, usecols=["MD", "Z", "GR", "TVT_input"])
    if "TVT" in frame.columns or set(frame.columns) != {"MD", "Z", "GR", "TVT_input"}:
        raise ValueError("target-free horizontal loader exposed an unexpected column")
    return frame


def validate_series_raw_parity(
    well: str,
    horizontal: pd.DataFrame,
    horizontal_series: pd.DataFrame,
    typewell_series: pd.DataFrame,
) -> None:
    if len(horizontal) != len(horizontal_series):
        raise ValueError(f"well={well} raw horizontal and exp304 series lengths differ")
    md = pd.to_numeric(horizontal["MD"], errors="raise").to_numpy(np.float64)
    coordinate = horizontal_series["coordinate"].to_numpy(np.float64)
    if not np.array_equal(md, coordinate):
        raise ValueError(f"well={well} exp304 horizontal coordinate does not equal raw MD")
    type_raw = typewell_series["raw_gr"].to_numpy(np.float64)
    fill = float(np.mean(type_raw))
    expected = (
        pd.to_numeric(horizontal["GR"], errors="coerce")
        .interpolate(limit_direction="both")
        .fillna(fill)
        .to_numpy(np.float64)
    )
    actual = horizontal_series["raw_gr"].to_numpy(np.float64)
    if not np.allclose(expected, actual, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"well={well} exp304 raw horizontal series parity failed")


# %% [markdown]
# ## 5. Tempered emission and exact exp209 forward-backward kernel


# %%
def build_tempered_emission(
    observed_raw_gr: np.ndarray,
    state_raw_gr: np.ndarray,
    observed_swt_gr: np.ndarray,
    state_swt_gr: np.ndarray,
    sigma: float,
    *,
    raw_weight: float = 0.85,
    swt_weight: float = 0.15,
    log_likelihood_clip: float = 600.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the one preregistered log-emission mixture with one raw sigma."""
    observed_raw = np.asarray(observed_raw_gr, dtype=np.float64)
    state_raw = np.asarray(state_raw_gr, dtype=np.float64)
    observed_swt = np.asarray(observed_swt_gr, dtype=np.float64)
    state_swt = np.asarray(state_swt_gr, dtype=np.float64)
    if observed_raw.ndim != 1 or observed_swt.ndim != 1 or len(observed_raw) != len(observed_swt):
        raise ValueError("raw and SWT observations must be aligned one-dimensional arrays")
    if state_raw.ndim != 1 or state_swt.ndim != 1 or len(state_raw) != len(state_swt):
        raise ValueError("raw and SWT state curves must be aligned one-dimensional arrays")
    if not (
        np.isfinite(observed_raw).all()
        and np.isfinite(observed_swt).all()
        and np.isfinite(state_raw).all()
        and np.isfinite(state_swt).all()
        and np.isfinite(sigma)
        and float(sigma) > 0.0
    ):
        raise ValueError("tempered emission requires finite signals and positive sigma")
    if float(raw_weight) != 0.85 or float(swt_weight) != 0.15:
        raise ValueError("exp305 permits only raw_weight=0.85 and swt_weight=0.15")
    raw_z = (observed_raw[:, None] - state_raw[None, :]) / float(sigma)
    swt_z = (observed_swt[:, None] - state_swt[None, :]) / float(sigma)
    ell_raw = -0.5 * np.minimum(raw_z**2, float(log_likelihood_clip))
    ell_swt = -0.5 * np.minimum(swt_z**2, float(log_likelihood_clip))
    ell_beta = float(raw_weight) * ell_raw + float(swt_weight) * ell_swt
    return (
        ell_beta.astype(np.float32),
        ell_raw.astype(np.float32),
        ell_swt.astype(np.float32),
    )


def robust_initial_rate(
    known_prefix: pd.DataFrame,
    window_rows: int = 30,
    *,
    min_valid_steps: int = 3,
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
        return 0.0, int(len(tail)), valid_steps
    rate = float(np.median((dtvt[valid] + dz[valid]) / dmd[valid]))
    return (rate if np.isfinite(rate) else 0.0), int(len(tail)), valid_steps


def prepare_tempered_hmm_inputs(
    horizontal: pd.DataFrame,
    horizontal_series: pd.DataFrame,
    typewell_series: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Prepare emissions without accepting unknown-suffix true TVT."""
    if "TVT" in horizontal.columns:
        raise ValueError("tempered HMM preparation forbids horizontal true TVT")
    hmm = fixed_hmm_kwargs(config)
    emission = get_nested(config, "model.emission") or {}
    known_mask = horizontal["TVT_input"].notna().to_numpy()
    eval_mask = ~known_mask
    known = horizontal.loc[known_mask]
    eval_rows = horizontal.loc[eval_mask]
    if len(known) < 4 or len(eval_rows) == 0:
        raise ValueError("each well requires at least four prefix and one suffix row")
    raw_type_tvt = typewell_series["coordinate"].to_numpy(np.float64)
    raw_type_gr = typewell_series["raw_gr"].to_numpy(np.float64)
    swt_type_gr = typewell_series["swt_db4_l3_gr"].to_numpy(np.float64)
    if bool((np.diff(raw_type_tvt) < 0.0).any()):
        raise ValueError("exp304 typewell coordinate must be non-decreasing")
    known_tvt = pd.to_numeric(known["TVT_input"], errors="raise").to_numpy(np.float64)
    typewell_at_known = np.interp(known_tvt, raw_type_tvt, raw_type_gr)
    residual = (
        pd.to_numeric(known["GR"], errors="coerce").fillna(0.0).to_numpy(np.float64)
        - typewell_at_known
    )
    sigma_low, sigma_high = [float(value) for value in emission["sigma_clip"]]
    gr_sigma = float(np.clip(np.nanstd(residual), sigma_low, sigma_high))
    if not np.isfinite(gr_sigma):
        raise ValueError("known-prefix raw-GR residual sigma is non-finite")
    init_rate, rate_rows, valid_steps = robust_initial_rate(known, 30)
    last = known.iloc[-1]
    last_tvt = float(last["TVT_input"])
    step = float(hmm["step"])
    grid_min = max(float(raw_type_tvt.min()) - 40.0, last_tvt - float(hmm["band_pad"]))
    grid_max = min(float(raw_type_tvt.max()) + 40.0, last_tvt + float(hmm["band_pad"]))
    grid = np.arange(grid_min, grid_max + step, step, dtype=np.float64)
    raw_grid = np.interp(grid, raw_type_tvt, raw_type_gr)
    swt_grid = np.interp(grid, raw_type_tvt, swt_type_gr)
    eval_index = np.flatnonzero(eval_mask).astype(np.int64)
    observed_raw = horizontal_series["raw_gr"].to_numpy(np.float64)[eval_index]
    observed_swt = horizontal_series["swt_db4_l3_gr"].to_numpy(np.float64)[eval_index]
    emission_ll, ell_raw, ell_swt = build_tempered_emission(
        observed_raw,
        raw_grid,
        observed_swt,
        swt_grid,
        gr_sigma,
        raw_weight=float(emission["raw_weight"]),
        swt_weight=float(emission["swt_weight"]),
        log_likelihood_clip=float(emission["log_likelihood_clip"]),
    )
    md = pd.to_numeric(eval_rows["MD"], errors="raise").to_numpy(np.float64)
    z = pd.to_numeric(eval_rows["Z"], errors="raise").to_numpy(np.float64)
    dm = np.maximum(np.diff(np.concatenate([[float(last["MD"])], md])), 1.0)
    dz = np.diff(np.concatenate([[float(last["Z"])], z]))
    span = max(float(hmm["rate_span"]), abs(init_rate) + 0.04)
    rates = np.linspace(-span, span, int(hmm["n_rates"]), dtype=np.float64)
    return {
        "emission_ll": emission_ll,
        "ell_raw": ell_raw,
        "ell_swt": ell_swt,
        "dm": dm,
        "dz": dz,
        "grid": grid,
        "rates": rates,
        "start_p": float((last_tvt - grid_min) / step),
        "r0": float(init_rate),
        "eval_index": eval_index,
        "last_known_tvt": last_tvt,
        "last_known_md": float(last["MD"]),
        "prefix_rows": int(len(known)),
        "prefix_sigma": gr_sigma,
        "prefix_ir": init_rate,
        "initial_rate_effective_rows": int(rate_rows),
        "initial_rate_valid_steps": int(valid_steps),
    }


@njit(cache=True, nogil=True, parallel=True)
def _hmm2_fb(
    em,
    dm,
    dz,
    sp,
    rates,
    sig_r,
    sig_p,
    start_p,
    start_sig,
    r0,
    r0_sig,
    lam,
    mom,
):
    """Amerhu/exp209 exact forward-backward over (TVT position, dip-rate)."""
    t_count, p_count = em.shape
    r_count = len(rates)
    rate_step = rates[1] - rates[0]
    neg = np.float32(-1e18)
    alpha = np.full((t_count, p_count, r_count), neg, np.float32)
    prev = np.full((p_count, r_count), neg, np.float32)
    for p_i in range(p_count):
        dpos = (p_i - start_p) * sp
        lp0 = -0.5 * (dpos / start_sig) ** 2
        if lp0 < -60.0:
            continue
        for r_i in range(r_count):
            dr = (rates[r_i] - r0) / r0_sig
            prev[p_i, r_i] = np.float32(lp0 - 0.5 * dr * dr)
    tmp = np.empty((p_count, r_count), np.float32)
    cur = np.empty((p_count, r_count), np.float32)

    for t_i in range(t_count):
        sig_rate_step = sig_r * np.sqrt(dm[t_i])
        rate_var_cells = (sig_rate_step / rate_step) ** 2
        rate_log_kernel = np.empty((r_count, 3))
        for r_i in range(r_count):
            mean_rate_move = -(1.0 - mom) * rates[r_i] * dm[t_i] / rate_step
            p_plus = max(0.5 * (rate_var_cells + mean_rate_move), 1e-12)
            p_minus = max(0.5 * (rate_var_cells - mean_rate_move), 1e-12)
            total = p_plus + p_minus
            if total > 0.9:
                p_plus *= 0.9 / total
                p_minus *= 0.9 / total
            rate_log_kernel[r_i, 0] = np.log(p_minus)
            rate_log_kernel[r_i, 1] = np.log(1.0 - p_plus - p_minus)
            rate_log_kernel[r_i, 2] = np.log(p_plus)
        for p_i in prange(p_count):
            for r2 in range(r_count):
                best = neg
                k0 = max(r2 - 1, 0)
                k1 = min(r2 + 1, r_count - 1)
                for r_i in range(k0, k1 + 1):
                    value = prev[p_i, r_i] + rate_log_kernel[r_i, r2 - r_i + 1]
                    if value > best:
                        best = value
                if best > neg / 2:
                    total = 0.0
                    for r_i in range(k0, k1 + 1):
                        total += np.exp(prev[p_i, r_i] + rate_log_kernel[r_i, r2 - r_i + 1] - best)
                    tmp[p_i, r2] = np.float32(best + np.log(total))
                else:
                    tmp[p_i, r2] = neg

        sigma_position = max(sig_p, 0.35 * sp)
        for r2 in prange(r_count):
            mu = rates[r2] * dm[t_i] - dz[t_i]
            b0 = int(np.floor(mu / sp + 0.5))
            position_log_kernel = np.empty(5)
            for k_i in range(5):
                delta = (b0 - 2 + k_i) * sp - mu
                position_log_kernel[k_i] = -0.5 * (delta / sigma_position) ** 2
            kernel_max = np.max(position_log_kernel)
            log_norm = kernel_max + np.log(np.sum(np.exp(position_log_kernel - kernel_max)))
            position_log_kernel -= log_norm
            for p2 in range(p_count):
                best = neg
                for k_i in range(5):
                    p1 = p2 - (b0 - 2 + k_i)
                    if 0 <= p1 < p_count:
                        value = tmp[p1, r2] + position_log_kernel[k_i]
                        if value > best:
                            best = value
                if best > neg / 2:
                    total = 0.0
                    for k_i in range(5):
                        p1 = p2 - (b0 - 2 + k_i)
                        if 0 <= p1 < p_count:
                            total += np.exp(tmp[p1, r2] + position_log_kernel[k_i] - best)
                    cur[p2, r2] = np.float32(best + np.log(total) + lam * em[t_i, p2])
                else:
                    cur[p2, r2] = neg
        for p_i in range(p_count):
            for r_i in range(r_count):
                alpha[t_i, p_i, r_i] = cur[p_i, r_i]
                prev[p_i, r_i] = cur[p_i, r_i]

    best = np.float32(neg)
    for p_i in range(p_count):
        for r_i in range(r_count):
            if alpha[t_count - 1, p_i, r_i] > best:
                best = alpha[t_count - 1, p_i, r_i]
    total = 0.0
    for p_i in range(p_count):
        for r_i in range(r_count):
            total += np.exp(alpha[t_count - 1, p_i, r_i] - best)
    loglik = float(best) + np.log(total)
    post_p = np.zeros((t_count, p_count))
    beta_next = np.zeros((p_count, r_count), np.float32)
    values = alpha[t_count - 1] + beta_next
    best = np.max(values)
    total = 0.0
    for p_i in range(p_count):
        acc = 0.0
        for r_i in range(r_count):
            acc += np.exp(values[p_i, r_i] - best)
        post_p[t_count - 1, p_i] = acc
        total += acc
    post_p[t_count - 1] /= total

    beta_cur = np.empty((p_count, r_count), np.float32)
    beta_tmp = np.empty((p_count, r_count), np.float32)
    for t_i in range(t_count - 1, 0, -1):
        sig_rate_step = sig_r * np.sqrt(dm[t_i])
        rate_var_cells = (sig_rate_step / rate_step) ** 2
        rate_log_kernel = np.empty((r_count, 3))
        for r_i in range(r_count):
            mean_rate_move = -(1.0 - mom) * rates[r_i] * dm[t_i] / rate_step
            p_plus = max(0.5 * (rate_var_cells + mean_rate_move), 1e-12)
            p_minus = max(0.5 * (rate_var_cells - mean_rate_move), 1e-12)
            total = p_plus + p_minus
            if total > 0.9:
                p_plus *= 0.9 / total
                p_minus *= 0.9 / total
            rate_log_kernel[r_i, 0] = np.log(p_minus)
            rate_log_kernel[r_i, 1] = np.log(1.0 - p_plus - p_minus)
            rate_log_kernel[r_i, 2] = np.log(p_plus)
        sigma_position = max(sig_p, 0.35 * sp)
        for r2 in prange(r_count):
            mu = rates[r2] * dm[t_i] - dz[t_i]
            b0 = int(np.floor(mu / sp + 0.5))
            position_log_kernel = np.empty(5)
            for k_i in range(5):
                delta = (b0 - 2 + k_i) * sp - mu
                position_log_kernel[k_i] = -0.5 * (delta / sigma_position) ** 2
            kernel_max = np.max(position_log_kernel)
            log_norm = kernel_max + np.log(np.sum(np.exp(position_log_kernel - kernel_max)))
            position_log_kernel -= log_norm
            for p1 in range(p_count):
                best = neg
                for k_i in range(5):
                    p2 = p1 + (b0 - 2 + k_i)
                    if 0 <= p2 < p_count:
                        value = position_log_kernel[k_i] + lam * em[t_i, p2] + beta_next[p2, r2]
                        if value > best:
                            best = value
                if best > neg / 2:
                    total = 0.0
                    for k_i in range(5):
                        p2 = p1 + (b0 - 2 + k_i)
                        if 0 <= p2 < p_count:
                            total += np.exp(
                                position_log_kernel[k_i]
                                + lam * em[t_i, p2]
                                + beta_next[p2, r2]
                                - best
                            )
                    beta_tmp[p1, r2] = np.float32(best + np.log(total))
                else:
                    beta_tmp[p1, r2] = neg
        for p_i in prange(p_count):
            for r_i in range(r_count):
                best = neg
                k0 = max(r_i - 1, 0)
                k1 = min(r_i + 1, r_count - 1)
                for r2 in range(k0, k1 + 1):
                    value = rate_log_kernel[r_i, r2 - r_i + 1] + beta_tmp[p_i, r2]
                    if value > best:
                        best = value
                if best > neg / 2:
                    total = 0.0
                    for r2 in range(k0, k1 + 1):
                        total += np.exp(
                            rate_log_kernel[r_i, r2 - r_i + 1] + beta_tmp[p_i, r2] - best
                        )
                    beta_cur[p_i, r_i] = np.float32(best + np.log(total))
                else:
                    beta_cur[p_i, r_i] = neg
        values = alpha[t_i - 1] + beta_cur
        best = np.max(values)
        total = 0.0
        for p_i in range(p_count):
            acc = 0.0
            for r_i in range(r_count):
                acc += np.exp(values[p_i, r_i] - best)
            post_p[t_i - 1, p_i] = acc
            total += acc
        post_p[t_i - 1] /= total
        for p_i in range(p_count):
            for r_i in range(r_count):
                beta_next[p_i, r_i] = beta_cur[p_i, r_i]
    return post_p, loglik


def run_tempered_hmm(prepared: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    hmm = fixed_hmm_kwargs(config)
    posterior, log_likelihood = _hmm2_fb(
        np.asarray(prepared["emission_ll"], dtype=np.float32),
        np.asarray(prepared["dm"], dtype=np.float64),
        np.asarray(prepared["dz"], dtype=np.float64),
        float(hmm["step"]),
        np.asarray(prepared["rates"], dtype=np.float64),
        float(hmm["sig_r"]),
        float(hmm["sig_p"]),
        float(prepared["start_p"]),
        float(hmm["start_sig"]),
        float(prepared["r0"]),
        float(hmm["r0_sig"]),
        float(hmm["lam"]),
        float(hmm["mom"]),
    )
    grid = np.asarray(prepared["grid"], dtype=np.float64)
    mean = posterior @ grid
    variance = posterior @ (grid**2) - mean**2
    standard_deviation = np.sqrt(np.maximum(variance, 0.0))
    if not (np.isfinite(mean).all() and np.isfinite(standard_deviation).all()):
        raise ValueError("tempered exact-HMM posterior output is non-finite")
    return {
        "mean": mean,
        "std": standard_deviation,
        "log_likelihood": float(log_likelihood),
        "posterior_row_sum_max_abs_error": float(np.max(np.abs(posterior.sum(axis=1) - 1.0))),
    }


# %% [markdown]
# ## 6. Target-free well decoding and prediction freeze


# %%
class DeterministicGzipCsvWriter:
    def __init__(self, path: Path, compresslevel: int = 6):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._raw = path.open("wb")
        self._gzip = gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=self._raw,
            compresslevel=int(compresslevel),
            mtime=0,
        )
        self._digest = hashlib.sha256()
        self._header_written = False
        self.rows = 0
        self.closed = False

    def append(self, frame: pd.DataFrame) -> None:
        if self.closed:
            raise RuntimeError("cannot append to a closed deterministic writer")
        payload = frame.to_csv(index=False, header=not self._header_written).encode()
        self._gzip.write(payload)
        self._digest.update(payload)
        self._header_written = True
        self.rows += len(frame)

    def close(self) -> dict[str, Any]:
        if not self.closed:
            self._gzip.close()
            self._raw.close()
            self.closed = True
        decompressed = self._digest.hexdigest()
        return {
            "path": str(self.path),
            "rows": self.rows,
            "bytes": self.path.stat().st_size,
            "raw_sha256": sha256_path(self.path),
            "decompressed_sha256": decompressed,
            "content_sha256": decompressed,
        }


def decode_tempered_well(
    well: str,
    series: pd.DataFrame,
    raw_dir: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.time()
    horizontal_series, typewell_series = validate_and_split_series(well, series)
    horizontal = load_raw_horizontal_without_truth(well, raw_dir)
    validate_series_raw_parity(well, horizontal, horizontal_series, typewell_series)
    prepared = prepare_tempered_hmm_inputs(
        horizontal,
        horizontal_series,
        typewell_series,
        config,
    )
    decoded = run_tempered_hmm(prepared, config)
    eval_index = np.asarray(prepared["eval_index"], dtype=np.int64)
    mean = np.asarray(decoded["mean"], dtype=np.float32)
    standard_deviation = np.asarray(decoded["std"], dtype=np.float32)
    prediction = pd.DataFrame(
        {
            "id": [f"{well}_{int(row_idx)}" for row_idx in eval_index],
            "well_id": str(well),
            "row_idx": eval_index,
            "tempered_hmm_tvt": mean,
            "tempered_hmm_std": standard_deviation,
        }
    )
    finite = bool(
        np.isfinite(prediction[["tempered_hmm_tvt", "tempered_hmm_std"]].to_numpy()).all()
    )
    if not finite:
        raise ValueError(f"well={well} produced non-finite tempered HMM output")
    meta = {
        "well_id": str(well),
        "status": "ok",
        "rows": len(prediction),
        "elapsed_seconds": time.time() - started,
        "grid_size": int(len(prepared["grid"])),
        "prefix_rows": int(prepared["prefix_rows"]),
        "prefix_sigma": float(prepared["prefix_sigma"]),
        "prefix_initial_rate": float(prepared["prefix_ir"]),
        "initial_rate_effective_rows": int(prepared["initial_rate_effective_rows"]),
        "initial_rate_valid_steps": int(prepared["initial_rate_valid_steps"]),
        "hmm_log_likelihood": float(decoded["log_likelihood"]),
        "posterior_row_sum_max_abs_error": float(decoded["posterior_row_sum_max_abs_error"]),
        "finite_coverage": 1.0,
        "silent_fallback_count": 0,
    }
    return prediction[PREDICTION_COLUMNS], meta


def generate_and_freeze_predictions(
    series_path: Path,
    raw_dir: Path,
    artifacts: Path,
    config: dict[str, Any],
    expected_wells: list[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    if not NUMBA_AVAILABLE:
        raise RuntimeError("numba is required for the exp209 exact-HMM kernel")
    requested_numba_threads = int(get_nested(config, "runtime.numba_num_threads"))
    set_num_threads(requested_numba_threads)
    outer_workers = int(get_nested(config, "runtime.outer_workers"))
    writer = DeterministicGzipCsvWriter(
        artifacts / f"{OUTPUT_PREFIX}_predictions.csv.gz",
        compresslevel=6,
    )
    metadata: list[dict[str, Any]] = []
    observed_wells: list[str] = []
    iterator = iter_exp304_well_series(series_path)

    def process(batch: list[tuple[str, pd.DataFrame]]) -> list[tuple[pd.DataFrame, dict[str, Any]]]:
        if outer_workers == 1:
            return [decode_tempered_well(well, frame, raw_dir, config) for well, frame in batch]
        try:
            from joblib import Parallel, delayed
        except ImportError as exception:
            raise RuntimeError("joblib is required for outer_workers=2") from exception
        return Parallel(n_jobs=outer_workers, prefer="threads")(
            delayed(decode_tempered_well)(well, frame, raw_dir, config) for well, frame in batch
        )

    batch: list[tuple[str, pd.DataFrame]] = []
    for item in iterator:
        batch.append(item)
        if len(batch) < outer_workers:
            continue
        results = process(batch)
        for (well, _), (prediction, meta) in zip(batch, results, strict=True):
            observed_wells.append(well)
            writer.append(prediction)
            metadata.append(meta)
            if len(observed_wells) % 25 == 0:
                print(
                    f"tempered exact-HMM wells={len(observed_wells)}/{len(expected_wells)}",
                    flush=True,
                )
        batch = []
    if batch:
        results = process(batch)
        for (well, _), (prediction, meta) in zip(batch, results, strict=True):
            observed_wells.append(well)
            writer.append(prediction)
            metadata.append(meta)
    frozen = writer.close()
    if observed_wells != expected_wells:
        raise ValueError("exp304 series well order/identity does not match exp226")
    expected_rows = int(get_nested(config, "validation.expected_rows"))
    if frozen["rows"] != expected_rows or len(metadata) != len(expected_wells):
        raise ValueError("tempered prediction row/well coverage mismatch")
    by_well_runtime = pd.DataFrame(metadata).sort_values("well_id", kind="mergesort")
    if (
        not bool((by_well_runtime["status"] == "ok").all())
        or int(by_well_runtime["silent_fallback_count"].sum()) != 0
        or float(by_well_runtime["finite_coverage"].min()) != 1.0
    ):
        raise ValueError("tempered HMM per-well technical contract failed")
    return frozen, by_well_runtime


# %% [markdown]
# ## 7. Late truth/control attachment and paired metrics
#
# The functions below are called only after the deterministic prediction gzip
# is closed and its decompressed/content SHA has been recorded.


# %%
def _require_frozen_prediction(frozen: dict[str, Any]) -> None:
    if not frozen.get("decompressed_sha256") or not frozen.get("content_sha256"):
        raise RuntimeError("truth/control attachment requires a frozen prediction content SHA")


def _assert_same_order(label: str, expected: pd.Series, actual: pd.Series) -> None:
    expected_values = expected.astype(str).to_numpy()
    actual_values = actual.astype(str).to_numpy()
    if len(expected_values) != len(actual_values) or not np.array_equal(
        expected_values, actual_values
    ):
        raise ValueError(f"{label} ID/order mismatch")


def materialize_saved_likpf_tvt(
    exp072: pd.DataFrame,
    control_spec: dict[str, Any],
) -> np.ndarray:
    column = str(control_spec["likpf_prediction_column"])
    values = pd.to_numeric(exp072[column], errors="raise").to_numpy(np.float64)
    representation = str(control_spec["likpf_prediction_representation"])
    if representation == "delta_from_last_known_tvt":
        anchor = pd.to_numeric(exp072["last_known_tvt"], errors="raise").to_numpy(
            np.float64
        )
        return anchor + values
    if representation == "absolute_tvt":
        return values
    raise ValueError(f"unsupported saved likPF representation: {representation}")


def load_late_readout_frame(
    preflight: dict[str, Any],
    frozen_prediction: dict[str, Any],
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require_frozen_prediction(frozen_prediction)
    prediction = pd.read_csv(
        frozen_prediction["path"],
        dtype={"id": str, "well_id": str},
    )
    prediction = prediction.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(
        drop=True
    )
    control_spec = get_nested(config, "data.saved_controls") or {}
    likpf_column = str(control_spec["likpf_prediction_column"])
    exp072 = pd.read_csv(
        preflight["controls"]["paths"]["saved_exp072"],
        usecols=["id", "well", "last_known_tvt", "md_since", likpf_column],
        dtype={"id": str, "well": str},
    )
    saved_hmm = pd.read_csv(
        preflight["controls"]["paths"]["saved_hmm"],
        usecols=["id", "well", str(control_spec["raw_hmm_prediction_column"])],
        dtype={"id": str, "well": str},
    )
    for control_frame in (exp072, saved_hmm):
        control_frame["_row_idx"] = pd.to_numeric(
            control_frame["id"].astype(str).str.rsplit("_", n=1).str[-1],
            errors="raise",
        ).astype(np.int64)
    exp072 = exp072.sort_values(["well", "_row_idx"], kind="mergesort").reset_index(drop=True)
    saved_hmm = saved_hmm.sort_values(["well", "_row_idx"], kind="mergesort").reset_index(drop=True)
    _assert_same_order("prediction vs exp072", prediction["id"], exp072["id"])
    _assert_same_order("prediction vs saved HMM", prediction["id"], saved_hmm["id"])
    fold_spec = get_nested(config, "data.fold_assignment") or {}
    truth_columns = [str(value) for value in fold_spec["truth_columns"]]
    truth = pd.read_csv(
        preflight["controls"]["paths"]["fold_assignment"],
        usecols=[*truth_columns, "fold"],
        dtype={"well_id": str},
    )
    truth["row_idx"] = pd.to_numeric(truth["row_idx"], errors="raise").astype(np.int64)
    truth["id"] = truth["well_id"].astype(str) + "_" + truth["row_idx"].astype(str)
    truth = truth.sort_values(["well_id", "row_idx"], kind="mergesort").reset_index(drop=True)
    _assert_same_order("prediction vs late truth", prediction["id"], truth["id"])
    hidden_spec = get_nested(config, "data.hidden_like_assignment") or {}
    role_columns = [str(value) for value in hidden_spec["role_columns"].values()]
    hidden = pd.read_csv(
        preflight["controls"]["paths"]["hidden_like_assignment"],
        usecols=["well_id", *role_columns],
        dtype={"well_id": str},
    )
    hidden = hidden.set_index("well_id")
    frame = pd.DataFrame(
        {
            "id": prediction["id"].astype(str),
            "well_id": prediction["well_id"].astype(str),
            "row_idx": prediction["row_idx"].to_numpy(np.int64),
            "fold": pd.to_numeric(truth["fold"], errors="raise").to_numpy(np.int64),
            "true_tvt": pd.to_numeric(truth["tvt_true"], errors="raise").to_numpy(np.float64),
            "md_since": pd.to_numeric(exp072["md_since"], errors="raise").to_numpy(np.float64),
            "tempered_hmm_tvt": pd.to_numeric(
                prediction["tempered_hmm_tvt"], errors="raise"
            ).to_numpy(np.float64),
            "raw_hmm_tvt": pd.to_numeric(
                saved_hmm[str(control_spec["raw_hmm_prediction_column"])], errors="raise"
            ).to_numpy(np.float64),
            "likpf_mean": materialize_saved_likpf_tvt(exp072, control_spec),
        }
    )
    for scope, role_column in hidden_spec["role_columns"].items():
        role_map = hidden[role_column].astype(str)
        frame[str(scope)] = frame["well_id"].map(role_map).eq("valid").to_numpy()
    frame["tempered_likpf_50_50"] = 0.5 * frame["tempered_hmm_tvt"] + 0.5 * frame["likpf_mean"]
    frame["raw_hmm_likpf_50_50"] = 0.5 * frame["raw_hmm_tvt"] + 0.5 * frame["likpf_mean"]
    numeric_columns = [
        "true_tvt",
        "md_since",
        "tempered_hmm_tvt",
        "raw_hmm_tvt",
        "likpf_mean",
        "tempered_likpf_50_50",
        "raw_hmm_likpf_50_50",
    ]
    if not np.isfinite(frame[numeric_columns].to_numpy(np.float64)).all():
        raise ValueError("late readout contains non-finite truth/control/prediction values")
    return frame, {
        "truth_attachment_stage": "after_tempered_prediction_gzip_and_content_sha_frozen",
        "prediction_content_sha256": frozen_prediction["content_sha256"],
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "id_mismatches": 0,
    }


def rmse(truth: np.ndarray, prediction: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    return float(np.sqrt(np.mean((prediction - truth) ** 2)))


def paired_metric_row(
    frame: pd.DataFrame,
    mask: np.ndarray,
    *,
    comparison: str,
    scope: str,
    candidate_column: str,
    control_column: str,
) -> dict[str, Any]:
    if not bool(mask.any()):
        raise ValueError(f"paired scope {scope} selected zero rows")
    truth = frame.loc[mask, "true_tvt"].to_numpy(np.float64)
    candidate = frame.loc[mask, candidate_column].to_numpy(np.float64)
    control = frame.loc[mask, control_column].to_numpy(np.float64)
    candidate_rmse = rmse(truth, candidate)
    control_rmse = rmse(truth, control)
    return {
        "comparison": comparison,
        "scope": scope,
        "rows": int(mask.sum()),
        "wells": int(frame.loc[mask, "well_id"].nunique()),
        "candidate_column": candidate_column,
        "control_column": control_column,
        "candidate_rmse": candidate_rmse,
        "control_rmse": control_rmse,
        "delta_rmse_candidate_minus_control": candidate_rmse - control_rmse,
        "improvement_ft": control_rmse - candidate_rmse,
    }


def build_paired_metrics(
    frame: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparisons = {
        "direct": ("tempered_hmm_tvt", "raw_hmm_tvt"),
        "blend": ("tempered_likpf_50_50", "raw_hmm_likpf_50_50"),
    }
    scopes: list[tuple[str, np.ndarray]] = [("overall", np.ones(len(frame), dtype=bool))]
    for fold in [int(value) for value in get_nested(config, "validation.expected_folds")]:
        scopes.append((f"fold_{fold}", frame["fold"].to_numpy(np.int64) == fold))
    scopes.extend(
        [
            ("md_since_1000_plus", frame["md_since"].to_numpy(np.float64) >= 1000.0),
            ("hidden_like_spatial", frame["hidden_like_spatial"].to_numpy(bool)),
            (
                "hidden_like_typewell_purged",
                frame["hidden_like_typewell_purged"].to_numpy(bool),
            ),
        ]
    )
    rows: list[dict[str, Any]] = []
    for comparison, (candidate_column, control_column) in comparisons.items():
        for scope, mask in scopes:
            rows.append(
                paired_metric_row(
                    frame,
                    mask,
                    comparison=comparison,
                    scope=scope,
                    candidate_column=candidate_column,
                    control_column=control_column,
                )
            )
    by_well_rows: list[dict[str, Any]] = []
    for well, group in frame.groupby("well_id", sort=True):
        truth = group["true_tvt"].to_numpy(np.float64)
        for comparison, (candidate_column, control_column) in comparisons.items():
            candidate_rmse = rmse(truth, group[candidate_column].to_numpy(np.float64))
            control_rmse = rmse(truth, group[control_column].to_numpy(np.float64))
            by_well_rows.append(
                {
                    "comparison": comparison,
                    "well_id": str(well),
                    "rows": len(group),
                    "candidate_rmse": candidate_rmse,
                    "control_rmse": control_rmse,
                    "delta_rmse_candidate_minus_control": candidate_rmse - control_rmse,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(by_well_rows)


# %% [markdown]
# ## 8. Promotion gate and generated artifacts


# %%
def evaluate_promotion_gate(
    paired_metrics: pd.DataFrame,
    by_well: pd.DataFrame,
    frame: pd.DataFrame,
    by_well_runtime: pd.DataFrame,
    preflight: dict[str, Any],
    runtime_seconds: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    promotion = get_nested(config, "audit.promotion") or {}
    baselines = get_nested(config, "audit.saved_baselines") or {}
    tolerance = float(promotion["non_degradation_float_tolerance_ft"])
    baseline_tolerance = float(baselines["metric_absolute_tolerance"])
    overall = paired_metrics.loc[paired_metrics["scope"] == "overall"].set_index("comparison")
    raw_hmm_actual = float(overall.loc["direct", "control_rmse"])
    blend_actual = float(overall.loc["blend", "control_rmse"])
    likpf_actual = rmse(
        frame["true_tvt"].to_numpy(np.float64),
        frame["likpf_mean"].to_numpy(np.float64),
    )
    baseline_parity = {
        "raw_hmm": {
            "actual": raw_hmm_actual,
            "expected": float(baselines["raw_hmm_rmse"]),
            "absolute_difference": abs(raw_hmm_actual - float(baselines["raw_hmm_rmse"])),
        },
        "saved_likpf": {
            "actual": likpf_actual,
            "expected": float(baselines["saved_likpf_rmse"]),
            "absolute_difference": abs(likpf_actual - float(baselines["saved_likpf_rmse"])),
        },
        "raw_hmm_likpf_50_50": {
            "actual": blend_actual,
            "expected": float(baselines["raw_hmm_likpf_50_50_rmse"]),
            "absolute_difference": abs(blend_actual - float(baselines["raw_hmm_likpf_50_50_rmse"])),
        },
    }
    for record in baseline_parity.values():
        record["passed"] = bool(record["absolute_difference"] <= baseline_tolerance)
    technical = {
        "input_preflight_passed": True,
        "exp304_selected_denoiser": preflight["exp304"]["selected_denoiser"],
        "exp304_silent_fallback_count": int(preflight["exp304"]["silent_fallback_count"]),
        "prediction_rows": len(frame),
        "prediction_wells": int(frame["well_id"].nunique()),
        "finite_coverage": float(
            np.isfinite(frame[["tempered_hmm_tvt", "tempered_likpf_50_50"]].to_numpy()).mean()
        ),
        "id_mismatches": 0,
        "hmm_well_runs": len(by_well_runtime),
        "silent_fallback_count": int(by_well_runtime["silent_fallback_count"].sum()),
        "runtime_seconds": runtime_seconds,
        "runtime_limit_seconds": float(get_nested(config, "runtime.runtime_limit_seconds")),
        "baseline_metric_parity": baseline_parity,
    }
    technical["passed"] = bool(
        technical["exp304_silent_fallback_count"] == 0
        and technical["prediction_rows"] == int(get_nested(config, "validation.expected_rows"))
        and technical["prediction_wells"] == int(get_nested(config, "validation.expected_wells"))
        and technical["finite_coverage"] == 1.0
        and technical["id_mismatches"] == int(promotion["id_mismatch_max"])
        and technical["hmm_well_runs"] == int(get_nested(config, "model.hmm.hmm_well_runs"))
        and technical["silent_fallback_count"] == 0
        and runtime_seconds <= technical["runtime_limit_seconds"]
        and all(bool(record["passed"]) for record in baseline_parity.values())
    )
    required_scopes = [str(value) for value in promotion["required_non_degradation_scopes"]]
    comparison_gates: dict[str, Any] = {}
    for comparison in ("direct", "blend"):
        selected = paired_metrics.loc[paired_metrics["comparison"] == comparison]
        selected_overall = selected.loc[selected["scope"] == "overall"].iloc[0]
        folds = selected.loc[selected["scope"].str.startswith("fold_")]
        folds_improved = int((folds["delta_rmse_candidate_minus_control"] < -tolerance).sum())
        scope_checks = {
            scope: bool(
                selected.loc[
                    selected["scope"] == scope,
                    "delta_rmse_candidate_minus_control",
                ].iloc[0]
                <= tolerance
            )
            for scope in required_scopes
            if scope != "by_well_rmse_p95"
        }
        well_rows = by_well.loc[by_well["comparison"] == comparison]
        candidate_p95 = float(well_rows["candidate_rmse"].quantile(0.95))
        control_p95 = float(well_rows["control_rmse"].quantile(0.95))
        p95_delta = candidate_p95 - control_p95
        scope_checks["by_well_rmse_p95"] = bool(p95_delta <= tolerance)
        worst_delta = float(well_rows["delta_rmse_candidate_minus_control"].max())
        minimum_improvement = float(
            promotion[
                "direct_minimum_rmse_improvement_ft"
                if comparison == "direct"
                else "blend_minimum_rmse_improvement_ft"
            ]
        )
        record = {
            "candidate_rmse": float(selected_overall["candidate_rmse"]),
            "control_rmse": float(selected_overall["control_rmse"]),
            "improvement_ft": float(selected_overall["improvement_ft"]),
            "minimum_improvement_ft": minimum_improvement,
            "folds_improved": folds_improved,
            "minimum_folds_improved": int(promotion["minimum_folds_improved"]),
            "required_scope_checks": scope_checks,
            "by_well_candidate_rmse_p95": candidate_p95,
            "by_well_control_rmse_p95": control_p95,
            "by_well_rmse_p95_delta": p95_delta,
            "worst_well_rmse_delta": worst_delta,
            "worst_well_rmse_delta_max": float(promotion["worst_well_delta_rmse_max_ft"]),
        }
        record["passed"] = bool(
            record["improvement_ft"] >= minimum_improvement
            and folds_improved >= int(promotion["minimum_folds_improved"])
            and all(scope_checks.values())
            and worst_delta <= float(promotion["worst_well_delta_rmse_max_ft"])
        )
        comparison_gates[comparison] = record
    passed = bool(
        technical["passed"]
        and comparison_gates["direct"]["passed"]
        and comparison_gates["blend"]["passed"]
    )
    return {
        "experiment": EXPERIMENT_NAME,
        "passed": passed,
        "decision": (
            "tempered_exact_hmm_train_side_gate_passed_no_automatic_inference"
            if passed
            else "close_without_rescue_and_keep_reserved_pf_transfer_closed"
        ),
        "technical_gate": technical,
        "comparison_gates": comparison_gates,
        "failure_action": promotion["failure_action"],
    }


def output_file_reports(paths: dict[str, Path]) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for name, path in paths.items():
        reports[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "raw_sha256": sha256_path(path),
        }
    return reports


def run_full_experiment(config: dict[str, Any]) -> dict[str, Any]:
    if not KAGGLE_WORKING_ROOT.exists() and os.environ.get("EXPERIMENT_ALLOW_LOCAL") != "1":
        raise RuntimeError(
            "Full exp305 must run first on Kaggle; local execution requires explicit smoke approval"
        )
    validate_scientific_contract(config, require_run_approval=True)
    started = time.time()
    artifacts = artifact_dir()
    raw_dir = train_data_dir(config)
    exp304_preflight = preflight_exp304(config)
    raw_preflight = validate_raw_well_identity(config, raw_dir)
    control_preflight = preflight_controls_and_assignments(config)
    expected_wells = control_preflight["fold_assignment"].pop("well_ids")
    preflight = {
        "experiment": EXPERIMENT_NAME,
        "exp304": exp304_preflight,
        "raw_train": raw_preflight,
        "controls": control_preflight,
        "truth_attached": False,
    }
    scientific_contract = build_scientific_contract(config)
    contract_path = artifacts / f"{OUTPUT_PREFIX}_scientific_contract.json"
    manifest_path = artifacts / f"{OUTPUT_PREFIX}_input_control_manifest.json"
    write_json(contract_path, scientific_contract)
    write_json(manifest_path, preflight)

    frozen_prediction, by_well_runtime = generate_and_freeze_predictions(
        Path(exp304_preflight["paths"]["series"]),
        raw_dir,
        artifacts,
        config,
        expected_wells,
    )
    prediction_frozen_at_seconds = time.time() - started
    # Unknown-suffix truth and row-level saved controls are first parsed here.
    frame, late_attachment = load_late_readout_frame(preflight, frozen_prediction, config)
    paired_metrics, by_well_metrics = build_paired_metrics(frame, config)
    runtime_seconds = time.time() - started
    promotion_gate = evaluate_promotion_gate(
        paired_metrics,
        by_well_metrics,
        frame,
        by_well_runtime,
        preflight,
        runtime_seconds,
        config,
    )

    output_paths = {
        "by_well_runtime": artifacts / f"{OUTPUT_PREFIX}_by_well_runtime.csv",
        "overall_fold_scope_metrics": artifacts / f"{OUTPUT_PREFIX}_overall_fold_scope_metrics.csv",
        "by_well_metrics": artifacts / f"{OUTPUT_PREFIX}_by_well_metrics.csv",
        "promotion_gate": artifacts / f"{OUTPUT_PREFIX}_promotion_gate.json",
    }
    by_well_runtime.to_csv(output_paths["by_well_runtime"], index=False)
    paired_metrics.to_csv(output_paths["overall_fold_scope_metrics"], index=False)
    by_well_metrics.to_csv(output_paths["by_well_metrics"], index=False)
    write_json(output_paths["promotion_gate"], promotion_gate)
    status = (
        "train_side_tempered_exact_hmm_gate_passed_no_inference"
        if promotion_gate["passed"]
        else "train_side_tempered_exact_hmm_gate_failed_closed"
    )
    summary = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "route": get_nested(config, "experiment.route"),
        "runtime_seconds": runtime_seconds,
        "prediction_frozen_at_seconds": prediction_frozen_at_seconds,
        "rows": len(frame),
        "wells": int(frame["well_id"].nunique()),
        "active_scientific_variants": 1,
        "hmm_well_runs": len(by_well_runtime),
        "models": 0,
        "lightgbm_configs": 0,
        "trained_folds": 0,
        "pf_runs": 0,
        "beam_runs": 0,
        "boosters": 0,
        "control_reruns": 0,
        "scientific_contract_sha256": scientific_contract["scientific_contract_sha256"],
        "input_control_manifest_sha256": sha256_path(manifest_path),
        "prediction": frozen_prediction,
        "truth_attachment": late_attachment,
        "promotion_gate": promotion_gate,
        "runtime_versions": runtime_versions(),
        "kaggle": {
            "kernel_version": None,
            "kernel_version_recording": "record_from_kaggle_api_after_run",
            "kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "model_sha256": None,
        "submission_sha256": None,
    }
    summary_path = artifacts / f"{OUTPUT_PREFIX}_summary.json"
    write_json(summary_path, summary)
    summary["generated_files"] = output_file_reports(
        {
            **output_paths,
            "scientific_contract": contract_path,
            "input_control_manifest": manifest_path,
        }
    )
    write_json(summary_path, summary)
    overall = paired_metrics.loc[paired_metrics["scope"] == "overall"]
    metrics = {
        "experiment": EXPERIMENT_NAME,
        "status": status,
        "route": "pf_beam",
        "cv": (
            float(overall.loc[overall["comparison"] == "blend", "candidate_rmse"].iloc[0])
            if promotion_gate["passed"]
            else None
        ),
        "public_lb": None,
        "private_lb": None,
        "metric": "rmse",
        "overall": overall.to_dict(orient="records"),
        "promotion_gate": promotion_gate,
        "prediction_sha256": frozen_prediction,
        "model_sha256": None,
        "submission_sha256": None,
        "notes": "Train-side only; no raw-test prediction, inference, or submission is produced.",
    }
    write_json(metrics_output_path(), metrics)
    print(overall.to_string(index=False))
    print(json.dumps(to_jsonable(promotion_gate), indent=2, sort_keys=True))
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))
    return summary


# %% [markdown]
# ## 9. Setup, configuration, and contract preview

# %%
CONFIG: dict[str, Any] | None = None
if EXECUTE_NOTEBOOK:
    CONFIG = load_experiment_config()
    validate_scientific_contract(CONFIG)
    print(
        json.dumps(
            {
                "experiment": get_nested(CONFIG, "experiment.name"),
                "route": get_nested(CONFIG, "experiment.route"),
                "parent": get_nested(CONFIG, "lineage.parent"),
                "methodology_parent": get_nested(CONFIG, "lineage.methodology_parent"),
                "scientific_variants": get_nested(CONFIG, "model.active_scientific_variants"),
                "emission": get_nested(CONFIG, "model.emission"),
                "hmm": get_nested(CONFIG, "model.hmm"),
                "blend": get_nested(CONFIG, "model.blend"),
                "expected_rows": get_nested(CONFIG, "validation.expected_rows"),
                "expected_wells": get_nested(CONFIG, "validation.expected_wells"),
                "saved_fold_strata": get_nested(CONFIG, "validation.n_folds"),
                "hmm_well_runs": get_nested(CONFIG, "model.hmm.hmm_well_runs"),
                "lightgbm_configs": get_nested(CONFIG, "model.lightgbm_config_count"),
                "trained_folds": get_nested(CONFIG, "model.fold_training_count"),
                "boosters": get_nested(CONFIG, "model.booster_count"),
                "implementation_approved": get_nested(CONFIG, "execution.implementation_approved"),
                "kaggle_push_approved": get_nested(CONFIG, "execution.kaggle_push_approved"),
                "run_train": get_nested(CONFIG, "execution.run_train"),
                "inference_enabled": get_nested(CONFIG, "inference.enabled"),
                "create_submission": get_nested(CONFIG, "execution.create_submission"),
            },
            indent=2,
            sort_keys=True,
        )
    )


# %% [markdown]
# ## 10. Run the Kaggle CPU audit

# %%
if EXECUTE_NOTEBOOK:
    assert CONFIG is not None
    EXP305_SUMMARY = run_full_experiment(CONFIG)
