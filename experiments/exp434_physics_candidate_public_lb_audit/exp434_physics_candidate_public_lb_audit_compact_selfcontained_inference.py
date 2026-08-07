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
# # exp434 physics-candidate Public-LB audit — hidden-safe inference candidate
#
# exp263 Stage 1で固定した6 primitiveをraw competition testから再生成し、5つの
# float32 50:50 pairと固定50/25/25 blendを同じ順序で構成する。候補・式・batchは
# configで事前固定し、1 notebook versionにつき承認済みの1候補だけを
# `submission.csv`へ出力する。
#
# このcompact self-contained版は正規Notebookへの採用前候補である。学習、OOF再計算、
# selector、weight tuning、fallback補完は行わない。Kaggle package / run /
# competition submissionはconfigの個別承認flagがそろわない限りfail-closeする。

# %% [markdown]
# ## Contents
#
# 1. Imports
# 2. Frozen candidate and execution contract
# 3. Runtime, path, SHA, and identity helpers
# 4. Raw-test primitive generation helpers
# 5. Formula, equivalence-gate, and submission helpers
# 6. Setup and frozen configuration preflight
# 7. Trusted source and provenance resolution
# 8. Hidden-safe six-primitive regeneration
# 9. Twelve-candidate formula bank and equivalence gates
# 10. Selected candidate and submission generation
# 11. Metrics, SHA, manifests, and generated artifacts

# %% [markdown]
# ## 1. Imports

# %%
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from IPython.display import display

# %% [markdown]
# ## 2. Frozen candidate and execution contract

# %%
EXPERIMENT_NAME = "exp434_physics_candidate_public_lb_audit"
IMPORT_ONLY_ENV = "EXP434_IMPORT_ONLY"
KAGGLE_INPUT_ROOT = Path("/kaggle/input")

PRIMITIVE_IDS = (
    "exp226_k16",
    "selfgr_hmm_a070",
    "likpf_mean",
    "exact_hmm",
    "pf_ancc",
    "beam_mean",
)
PAIR_SPECS = (
    ("exp226_k16__selfgr_hmm_a070", "exp226_k16", "selfgr_hmm_a070"),
    ("exp226_k16__exact_hmm", "exp226_k16", "exact_hmm"),
    ("exp226_k16__likpf_mean", "exp226_k16", "likpf_mean"),
    ("selfgr_hmm_a070__likpf_mean", "selfgr_hmm_a070", "likpf_mean"),
    ("likpf_mean__exact_hmm", "likpf_mean", "exact_hmm"),
)
PAIR_IDS = tuple(spec[0] for spec in PAIR_SPECS)
FIXED_ID = "exp226_w500_50_50"
CANDIDATE_IDS = (*PRIMITIVE_IDS, *PAIR_IDS, FIXED_ID)
NORMAL_SUBMISSION_IDS = (
    *PAIR_IDS,
    "selfgr_hmm_a070",
    "exact_hmm",
    "pf_ancc",
    "beam_mean",
)
CONDITIONAL_EQUIVALENCE_IDS = ("exp226_k16", "likpf_mean")

EXPECTED_FORMULAS = {
    **{candidate_id: candidate_id for candidate_id in PRIMITIVE_IDS},
    **{
        pair_id: f"0.5*{left} + 0.5*{right}"
        for pair_id, left, right in PAIR_SPECS
    },
    FIXED_ID: "0.5*exp226_k16 + 0.25*likpf_mean + 0.25*exact_hmm",
}
EXPECTED_KINDS = {
    **{candidate_id: "primitive" for candidate_id in PRIMITIVE_IDS},
    **{candidate_id: "pair" for candidate_id in PAIR_IDS},
    FIXED_ID: "fixed",
}


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _formula_token(value: str) -> str:
    return "".join(str(value).split())


def candidate_entries(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = config.get("candidate_contract", {}).get("candidates")
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise ValueError("candidate_contract.candidates must be a list of mappings")
    return [dict(item) for item in raw]


def validate_candidate_contract(
    config: Mapping[str, Any],
    *,
    require_run_approval: bool = False,
) -> dict[str, Any]:
    experiment = config.get("experiment", {})
    contract = config.get("candidate_contract", {})
    model = config.get("model", {})
    plan = config.get("submission_plan", {})
    execution = config.get("execution", {})
    runtime = config.get("runtime", {})
    entries = candidate_entries(config)
    by_id = {str(item.get("id")): item for item in entries}

    if experiment.get("name") != EXPERIMENT_NAME:
        raise ValueError("experiment name differs from the frozen exp434 contract")
    if experiment.get("route") != "pf_beam":
        raise ValueError("exp434 route must remain pf_beam")
    if tuple(by_id) != CANDIDATE_IDS or len(entries) != len(by_id):
        raise ValueError("candidate order/inventory differs from the frozen twelve")
    if (
        int(contract.get("candidate_count", -1)) != 12
        or int(contract.get("primitive_count", -1)) != 6
        or int(contract.get("pair_count", -1)) != 5
        or int(contract.get("fixed_count", -1)) != 1
    ):
        raise ValueError("candidate kind counts differ from 6 primitive + 5 pair + 1 fixed")
    if str(contract.get("dtype")) != "float32":
        raise ValueError("candidate dtype must remain float32")
    if contract.get("pair_arithmetic") != "float32_0p5_left_plus_0p5_right":
        raise ValueError("pair arithmetic differs from the frozen float32 contract")
    if (
        contract.get("fixed_arithmetic")
        != "float32_0p5_k16_plus_0p25_likpf_plus_0p25_exact_hmm"
    ):
        raise ValueError("fixed arithmetic differs from the frozen contract")

    for candidate_id in CANDIDATE_IDS:
        item = by_id[candidate_id]
        if item.get("kind") != EXPECTED_KINDS[candidate_id]:
            raise ValueError(f"candidate kind changed: {candidate_id}")
        if _formula_token(item.get("formula", "")) != _formula_token(
            EXPECTED_FORMULAS[candidate_id]
        ):
            raise ValueError(f"candidate formula changed: {candidate_id}")
        oof_rmse = float(item.get("oof_rmse", np.nan))
        if not np.isfinite(oof_rmse) or oof_rmse <= 0.0:
            raise ValueError(f"candidate OOF provenance missing: {candidate_id}")

    if tuple(plan.get("batch_1", ())) != PAIR_IDS:
        raise ValueError("batch 1 must remain the five frozen pair candidates")
    if tuple(plan.get("batch_2", ())) != NORMAL_SUBMISSION_IDS[5:]:
        raise ValueError("batch 2 must remain the four frozen primitive candidates")
    if not bool(plan.get("frozen_before_scores")):
        raise ValueError("submission batches must remain frozen before scores")
    if not bool(plan.get("forbid_adaptation_between_batches")):
        raise ValueError("between-batch adaptation must remain forbidden")
    if int(plan.get("normal_new_submission_count", -1)) != 9:
        raise ValueError("normal submission count must remain nine")
    if int(plan.get("maximum_new_submission_count", -1)) != 11:
        raise ValueError("maximum submission count must remain eleven")

    if model.get("active_variants") != []:
        raise ValueError("exp434 must train no active model variant")
    for key in ("lightgbm_config_count", "fold_training_count", "booster_count"):
        if int(model.get(key, -1)) != 0:
            raise ValueError(f"exp434 {key} must remain zero")
    if bool(model.get("parent_control_retraining")):
        raise ValueError("exp434 must not retrain a parent/control")
    if bool(runtime.get("use_gpu")) or runtime.get("device") != "cpu":
        raise ValueError("exp434 runtime must remain CPU-only")
    if not bool(execution.get("implementation_approved")):
        raise RuntimeError("exp434 implementation is not approved")

    selected = execution.get("selected_candidate")
    if selected is not None:
        selected = str(selected)
        allowed = set(NORMAL_SUBMISSION_IDS)
        existing = config.get("existing_lb", {})
        for conditional in CONDITIONAL_EQUIVALENCE_IDS:
            if (
                existing.get(conditional, {}).get("status")
                == "equivalence_failed_submit_required"
            ):
                allowed.add(conditional)
        if selected not in allowed:
            raise ValueError(
                f"selected candidate is not approved by the frozen plan: {selected}"
            )
        if selected == FIXED_ID:
            raise ValueError("the exact parent fixed blend must not be resubmitted")

    if require_run_approval:
        required_true = (
            "canonical_notebook_adoption_approved",
            "kaggle_package_approved",
            "kaggle_push_approved",
            "kaggle_run_approved",
            "run_inference",
            "create_submission",
        )
        missing = [key for key in required_true if not bool(execution.get(key))]
        if missing:
            raise RuntimeError(
                "Kaggle inference/package/run is not approved: " + ", ".join(missing)
            )
        if selected is None:
            raise RuntimeError("one frozen selected_candidate is required")
        if not execution.get("candidate_version_label"):
            raise RuntimeError("candidate_version_label is required for a run")
        canonical_sha = str(execution.get("canonical_notebook_sha256") or "")
        if len(canonical_sha) != 64:
            raise RuntimeError("canonical_notebook_sha256 must be frozen before a run")

    scientific_contract = {
        "candidate_ids": list(CANDIDATE_IDS),
        "formulas": EXPECTED_FORMULAS,
        "kinds": EXPECTED_KINDS,
        "batch_1": list(PAIR_IDS),
        "batch_2": list(NORMAL_SUBMISSION_IDS[5:]),
        "normal_submission_count": 9,
        "maximum_submission_count": 11,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "parent_retraining": 0,
        "device": "cpu",
    }
    scientific_contract["sha256"] = canonical_json_sha256(scientific_contract)
    return scientific_contract


# %% [markdown]
# ## 3. Runtime, path, SHA, and identity helpers

# %%
def locate_package_dir() -> Path:
    cwd = Path.cwd()
    candidates = (
        cwd,
        cwd / "experiments" / EXPERIMENT_NAME,
        Path("/kaggle/working"),
    )
    for candidate in candidates:
        config_path = candidate / "config.yaml"
        if config_path.exists():
            loaded = yaml.safe_load(config_path.read_text()) or {}
            if loaded.get("experiment", {}).get("name") == EXPERIMENT_NAME:
                return candidate
    raise FileNotFoundError(f"could not locate {EXPERIMENT_NAME}/config.yaml")


PACKAGE_DIR = locate_package_dir()


def load_config() -> dict[str, Any]:
    value = yaml.safe_load((PACKAGE_DIR / "config.yaml").read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError("config.yaml must contain a mapping")
    return value


def nested(config: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return default
        value = value[part]
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


def frame_content_sha256(
    frame: pd.DataFrame,
    columns: tuple[str, ...] | list[str] | None = None,
) -> str:
    selected = frame if columns is None else frame[list(columns)]
    selected = _normalize_frame_for_hash(selected)
    digest = hashlib.sha256()
    digest.update("|".join(selected.columns).encode())
    digest.update("|".join(str(dtype) for dtype in selected.dtypes).encode())
    row_hashes = pd.util.hash_pandas_object(selected, index=False, categorize=True)
    digest.update(row_hashes.to_numpy(dtype="uint64").astype("<u8", copy=False).tobytes())
    return digest.hexdigest()


def schema_sha256(frame: pd.DataFrame) -> str:
    normalized = _normalize_frame_for_hash(frame)
    schema = [(column, str(dtype)) for column, dtype in normalized.dtypes.items()]
    return canonical_json_sha256(schema)


def source_record(
    path: Path,
    *,
    expected_sha256: str | None = None,
    gzip_content: bool = False,
    expected_content_sha256: str | None = None,
) -> dict[str, Any]:
    file_sha = sha256_file(path)
    if expected_sha256 is not None and file_sha != expected_sha256:
        raise ValueError(
            f"source SHA mismatch for {path.name}: {file_sha} != {expected_sha256}"
        )
    record: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha,
    }
    if gzip_content:
        content_sha = sha256_gzip_content(path)
        if (
            expected_content_sha256 is not None
            and content_sha != expected_content_sha256
        ):
            raise ValueError(
                f"decompressed source SHA mismatch for {path.name}: "
                f"{content_sha} != {expected_content_sha256}"
            )
        record["decompressed_content_sha256"] = content_sha
    return record


def resolve_unique_source(
    filename: str,
    path_token: str,
    *,
    root: Path = KAGGLE_INPUT_ROOT,
) -> Path:
    matches = [
        path for path in sorted(root.rglob(filename)) if path_token in str(path)
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one {filename} under token {path_token}, got {matches}"
        )
    return matches[0]


def copy_trusted_source(source: Path, target_dir: Path, module_name: str) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{module_name}.py"
    shutil.copy2(source, target)
    return target


def resolve_data_root() -> Path:
    candidates = (
        Path("/kaggle/input/competitions/rogii-wellbore-geology-prediction"),
        Path("/kaggle/input/rogii-wellbore-geology-prediction"),
        PACKAGE_DIR.parent.parent / "data" / "raw",
        PACKAGE_DIR / "data" / "raw",
    )
    for candidate in candidates:
        if (
            (candidate / "train").is_dir()
            and (candidate / "test").is_dir()
            and (candidate / "sample_submission.csv").is_file()
        ):
            return candidate
    if KAGGLE_INPUT_ROOT.exists():
        matches = sorted(KAGGLE_INPUT_ROOT.rglob("sample_submission.csv"))
        valid = [
            path.parent
            for path in matches
            if (path.parent / "train").is_dir() and (path.parent / "test").is_dir()
        ]
        if len(valid) == 1:
            return valid[0]
    raise FileNotFoundError("competition train/test/sample_submission root not found")


@dataclass(frozen=True)
class RuntimePaths:
    data_root: Path
    train_dir: Path
    test_dir: Path
    sample_submission: Path
    output_dir: Path
    artifacts_dir: Path
    submission: Path
    metrics: Path


def build_runtime_paths() -> RuntimePaths:
    if not KAGGLE_INPUT_ROOT.is_dir():
        raise RuntimeError("exp434 inference must run in a Kaggle notebook runtime")
    data_root = resolve_data_root()
    output_dir = Path("/kaggle/working")
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return RuntimePaths(
        data_root=data_root,
        train_dir=data_root / "train",
        test_dir=data_root / "test",
        sample_submission=data_root / "sample_submission.csv",
        output_dir=output_dir,
        artifacts_dir=artifacts_dir,
        submission=output_dir / "submission.csv",
        metrics=output_dir / "metrics.json",
    )


def parse_identity(frame: pd.DataFrame) -> pd.DataFrame:
    if "id" not in frame:
        raise ValueError("candidate frame lacks id")
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


def validate_sample_submission(sample: pd.DataFrame) -> dict[str, Any]:
    if list(sample.columns) != ["id", "tvt"]:
        raise ValueError(
            f"sample submission columns must be ['id', 'tvt'], got {list(sample.columns)}"
        )
    identity = parse_identity(sample)
    if identity["id"].duplicated().any():
        raise ValueError("sample submission contains duplicate ids")
    if identity.duplicated(["well", "well_row_idx"]).any():
        raise ValueError("sample submission contains duplicate well/row identity")
    return {
        "rows": len(identity),
        "wells": int(identity["well"].nunique()),
        "id_content_sha256": frame_content_sha256(identity),
        "schema_sha256": schema_sha256(sample),
    }


# %% [markdown]
# ## 4. Raw-test primitive generation helpers

# %%
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
        item = pd.DataFrame(
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
        if self_gr is not None:
            item["candidate_finite_source"] = np.isfinite(
                np.asarray(result["mean_eval"], dtype=np.float32)
            ).astype(np.float32)
            item["selfgr_quality"] = np.asarray(
                result["self_gr_quality"], dtype=np.float32
            )
            item["selfgr_peak_tvt"] = np.asarray(
                result["self_gr_peak_tvt"], dtype=np.float32
            )
            item["score_margin"] = np.asarray(
                result["self_gr_peak_gap"], dtype=np.float32
            )
            item["selfgr_typewell_agreement"] = np.asarray(
                result["self_gr_typewell_agreement"], dtype=np.float32
            )
            item["selfgr_valid"] = np.asarray(
                result["self_gr_valid"], dtype=np.float32
            )
        rows.append(item)
    if not rows:
        raise ValueError("HMM raw-test generation produced no rows")
    output = finalize_primitive_confidence(pd.concat(rows, ignore_index=True))
    if output.duplicated("id").any() or not np.isfinite(
        output["candidate_tvt"]
    ).all():
        raise ValueError("HMM raw-test output violates duplicate/finite contract")
    return output


def generate_k16_primitive(
    module: Any,
    *,
    train_dir: Path,
    test_dir: Path,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    params = module.params_from_config(config)
    max_train = nested(config, "inference.max_train_wells")
    max_test = nested(config, "inference.max_test_wells")
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
        raise ValueError("exp226 K16 output violates duplicate/finite contract")
    return output, {
        "experiment": str(nested(config, "experiment.name")),
        "train_wells": len(train_wells),
        "test_wells": len(test_wells),
        "rows": len(output),
        "kappa": [float(value) for value in np.asarray(kappa).ravel()],
        "well_summaries": well_summaries,
        "prediction_and_confidence_content_sha256": frame_content_sha256(output),
    }


def primitive_audit(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    if tuple(frames) != PRIMITIVE_IDS:
        raise ValueError("primitive generation must preserve the frozen six-item order")
    normalized: dict[str, pd.DataFrame] = {}
    audit: dict[str, dict[str, Any]] = {}
    identity_columns = ["id", "well", "well_row_idx"]
    for candidate_id in PRIMITIVE_IDS:
        frame = frames[candidate_id].copy()
        missing = set([*identity_columns, "candidate_tvt"]) - set(frame)
        if missing:
            raise ValueError(
                f"primitive columns missing for {candidate_id}: {sorted(missing)}"
            )
        frame["id"] = frame["id"].astype(str)
        frame["well"] = frame["well"].astype(str)
        frame["well_row_idx"] = pd.to_numeric(
            frame["well_row_idx"], errors="raise"
        ).astype(np.int32)
        frame["candidate_tvt"] = pd.to_numeric(
            frame["candidate_tvt"], errors="coerce"
        ).astype(np.float32)
        frame = frame.sort_values(
            ["well", "well_row_idx"], kind="stable"
        ).reset_index(drop=True)
        if frame["id"].duplicated().any() or frame.duplicated(
            ["well", "well_row_idx"]
        ).any():
            raise ValueError(f"duplicate primitive identity: {candidate_id}")
        if not np.isfinite(frame["candidate_tvt"]).all():
            raise ValueError(f"nonfinite primitive prediction: {candidate_id}")
        normalized[candidate_id] = frame
        audit[candidate_id] = {
            "rows": len(frame),
            "wells": int(frame["well"].nunique()),
            "finite_rows": int(np.isfinite(frame["candidate_tvt"]).sum()),
            "fallback_rows": 0,
            "content_sha256": frame_content_sha256(frame),
            "prediction_sha256": frame_content_sha256(
                frame[["id", "candidate_tvt"]].rename(
                    columns={"candidate_tvt": "prediction"}
                )
            ),
        }
    base = normalized[PRIMITIVE_IDS[0]][identity_columns]
    for candidate_id in PRIMITIVE_IDS[1:]:
        if not base.equals(normalized[candidate_id][identity_columns]):
            raise ValueError(f"primitive identity mismatch: {candidate_id}")
    return base.copy(), audit


# %% [markdown]
# ## 5. Formula, equivalence-gate, and submission helpers

# %%
def build_candidate_bank(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, float]]:
    identity, _ = primitive_audit(frames)
    bank = identity.copy()
    normalized = {
        candidate_id: frames[candidate_id]
        .sort_values(["well", "well_row_idx"], kind="stable")
        .reset_index(drop=True)
        for candidate_id in PRIMITIVE_IDS
    }
    for candidate_id in PRIMITIVE_IDS:
        bank[candidate_id] = normalized[candidate_id]["candidate_tvt"].to_numpy(
            np.float32
        )

    formula_parity: dict[str, float] = {}
    for pair_id, left, right in PAIR_SPECS:
        left_value = bank[left].to_numpy(np.float32)
        right_value = bank[right].to_numpy(np.float32)
        frozen = (
            np.float32(0.5) * left_value + np.float32(0.5) * right_value
        ).astype(np.float32)
        parent_equivalent = (
            np.float32(0.5) * (left_value + right_value)
        ).astype(np.float32)
        max_abs = float(
            np.abs(
                frozen.astype(np.float64) - parent_equivalent.astype(np.float64)
            ).max(initial=0.0)
        )
        if max_abs > 1.0e-5:
            raise ValueError(f"parent float32 pair parity failed: {pair_id}")
        bank[pair_id] = frozen
        formula_parity[pair_id] = max_abs

    fixed = (
        np.float32(0.5) * bank["exp226_k16"].to_numpy(np.float32)
        + np.float32(0.25) * bank["likpf_mean"].to_numpy(np.float32)
        + np.float32(0.25) * bank["exact_hmm"].to_numpy(np.float32)
    ).astype(np.float32)
    bank[FIXED_ID] = fixed
    direct = (
        np.float32(0.5) * bank["exp226_k16"].to_numpy(np.float32)
        + np.float32(0.25) * bank["likpf_mean"].to_numpy(np.float32)
        + np.float32(0.25) * bank["exact_hmm"].to_numpy(np.float32)
    ).astype(np.float32)
    fixed_max_abs = float(
        np.abs(fixed.astype(np.float64) - direct.astype(np.float64)).max(
            initial=0.0
        )
    )
    if fixed_max_abs != 0.0:
        raise ValueError("fixed float32 formula parity must be exact")
    formula_parity[FIXED_ID] = fixed_max_abs

    if tuple(bank.columns[3:]) != CANDIDATE_IDS:
        raise ValueError("candidate bank column order differs from the frozen manifest")
    if not np.isfinite(bank[list(CANDIDATE_IDS)].to_numpy()).all():
        raise ValueError("candidate bank contains nonfinite predictions")
    return bank, formula_parity


def build_submission(
    sample: pd.DataFrame,
    bank: pd.DataFrame,
    candidate_id: str,
) -> pd.DataFrame:
    if candidate_id not in CANDIDATE_IDS:
        raise ValueError(f"unknown frozen candidate: {candidate_id}")
    validate_sample_submission(sample)
    values = bank[["id", candidate_id]].copy()
    values["id"] = values["id"].astype(str)
    sample_ids = sample["id"].astype(str).reset_index(drop=True)
    if values["id"].duplicated().any():
        raise ValueError("candidate bank contains duplicate submission id")
    if len(sample) != len(values) or set(sample_ids) != set(values["id"]):
        raise ValueError("candidate bank IDs do not match sample submission")
    output = sample[["id"]].copy()
    output["id"] = sample_ids
    output = output.merge(
        values,
        on="id",
        how="left",
        sort=False,
        validate="one_to_one",
    ).rename(columns={candidate_id: "tvt"})
    output["tvt"] = pd.to_numeric(output["tvt"], errors="coerce")
    if output["tvt"].isna().any() or not np.isfinite(output["tvt"]).all():
        raise ValueError("submission contains missing/nonfinite tvt")
    if not output["id"].equals(sample_ids):
        raise ValueError("submission does not preserve sample order")
    return output[["id", "tvt"]]


REFERENCE_COLUMNS = {
    "exp226_k16": "exp226_v6_k16_geometry_gr_u_projection",
    "selfgr_hmm_a070": "hmm_selfgr_boost_only_a070_c100_mean_tvt",
    "likpf_mean": "likpf_mean",
    "exact_hmm": "hmm_exact_mean_tvt",
    "pf_ancc": "pf_ancc",
    "beam_mean": "beam_mean",
}


def compare_exposed_reference(
    bank: pd.DataFrame,
    primitive_frames: Mapping[str, pd.DataFrame],
    reference_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    reference_config = nested(config, "data.exposed_reference", {})
    source = source_record(
        reference_path,
        expected_sha256=str(reference_config["raw_sha256"]),
        gzip_content=True,
        expected_content_sha256=str(
            reference_config["decompressed_content_sha256"]
        ),
    )
    reference = pd.read_csv(
        reference_path,
        usecols=["id", *REFERENCE_COLUMNS.values()],
        dtype={"id": str},
        low_memory=False,
    ).rename(
        columns={
            reference_column: candidate_id
            for candidate_id, reference_column in REFERENCE_COLUMNS.items()
        }
    )
    generated_ids = set(bank["id"].astype(str))
    reference_ids = set(reference["id"].astype(str))
    if generated_ids != reference_ids:
        return {
            "status": "skipped_hidden_id_set_differs_from_exposed_reference",
            "generated_rows": len(bank),
            "reference_rows": len(reference),
            "source": source,
        }

    aligned = bank[["id", *PRIMITIVE_IDS]].merge(
        reference[["id", *PRIMITIVE_IDS]],
        on="id",
        how="left",
        validate="one_to_one",
        suffixes=("_generated", "_reference"),
    )
    tolerance = float(nested(config, "existing_lb.equivalence_max_abs_tolerance_ft"))
    max_abs: dict[str, float] = {}
    content_sha: dict[str, str] = {}
    expected_content = nested(config, "data.exposed_primitive_content_sha256", {})
    for candidate_id in PRIMITIVE_IDS:
        difference = np.abs(
            aligned[f"{candidate_id}_generated"].to_numpy(np.float64)
            - aligned[f"{candidate_id}_reference"].to_numpy(np.float64)
        )
        max_abs[candidate_id] = float(difference.max(initial=0.0))
        content_sha[candidate_id] = frame_content_sha256(
            primitive_frames[candidate_id]
        )
        if content_sha[candidate_id] != expected_content[candidate_id]:
            raise ValueError(
                f"exposed primitive content SHA changed: {candidate_id}"
            )
    failed = {key: value for key, value in max_abs.items() if value > tolerance}
    if failed:
        raise ValueError(f"exp263 exposed primitive parity failed: {failed}")
    return {
        "status": "passed",
        "tolerance_ft": tolerance,
        "max_abs_ft": max_abs,
        "primitive_content_sha256": content_sha,
        "source": source,
    }


def compare_parent_formula_bank(
    bank: pd.DataFrame,
    parent_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    source = source_record(
        parent_path,
        expected_sha256=str(nested(config, "data.parent_stage1_formula_parity_sha256")),
    )
    parent = pd.read_parquet(
        parent_path,
        columns=["id", *CANDIDATE_IDS],
    )
    generated_ids = set(bank["id"].astype(str))
    parent_ids = set(parent["id"].astype(str))
    if generated_ids != parent_ids:
        return {
            "status": "skipped_hidden_id_set_differs_from_parent_exposed_bank",
            "generated_rows": len(bank),
            "parent_rows": len(parent),
            "source": source,
        }
    aligned = bank[["id", *CANDIDATE_IDS]].merge(
        parent[["id", *CANDIDATE_IDS]],
        on="id",
        how="left",
        validate="one_to_one",
        suffixes=("_generated", "_parent"),
    )
    max_abs = {}
    for candidate_id in CANDIDATE_IDS:
        difference = np.abs(
            aligned[f"{candidate_id}_generated"].to_numpy(np.float64)
            - aligned[f"{candidate_id}_parent"].to_numpy(np.float64)
        )
        max_abs[candidate_id] = float(difference.max(initial=0.0))
    failed = {key: value for key, value in max_abs.items() if value > 0.001}
    if failed:
        raise ValueError(f"exp263 parent candidate-bank parity failed: {failed}")
    return {
        "status": "passed",
        "tolerance_ft": 0.001,
        "max_abs_ft": max_abs,
        "source": source,
    }


def existing_submission_gate(
    bank: pd.DataFrame,
    candidate_id: str,
    submission_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    existing = nested(config, f"existing_lb.{candidate_id}", {})
    source = source_record(
        submission_path,
        expected_sha256=str(existing["submission_sha256"]),
    )
    previous = pd.read_csv(submission_path, dtype={"id": str})
    if list(previous.columns) != ["id", "tvt"]:
        raise ValueError(
            f"existing {candidate_id} submission has unexpected columns: "
            f"{list(previous.columns)}"
        )
    if previous["id"].duplicated().any():
        raise ValueError(f"existing {candidate_id} submission has duplicate ids")
    generated_ids = set(bank["id"].astype(str))
    previous_ids = set(previous["id"].astype(str))
    if generated_ids != previous_ids:
        return {
            "status": "skipped_hidden_id_set_differs_from_existing_submission",
            "generated_rows": len(bank),
            "existing_rows": len(previous),
            "source": source,
        }
    aligned = bank[["id", candidate_id]].merge(
        previous,
        on="id",
        how="left",
        validate="one_to_one",
        suffixes=("_generated", "_existing"),
    )
    difference = np.abs(
        aligned[candidate_id].to_numpy(np.float64)
        - aligned["tvt"].to_numpy(np.float64)
    )
    max_abs = float(difference.max(initial=0.0))
    tolerance = float(
        nested(config, "existing_lb.equivalence_max_abs_tolerance_ft")
    )
    if max_abs > tolerance:
        status = (
            "failed_submit_same_candidate_required"
            if candidate_id in CONDITIONAL_EQUIVALENCE_IDS
            else "failed_exact_parent_submission_mismatch"
        )
        if candidate_id == FIXED_ID:
            raise ValueError(
                f"fixed parent submission exact-equivalence failed: {max_abs}"
            )
    else:
        status = (
            "passed_reusable_exact_parent_provenance"
            if candidate_id == FIXED_ID
            else "passed_reusable_existing_public_lb"
        )
    return {
        "status": status,
        "max_abs_ft": max_abs,
        "tolerance_ft": tolerance,
        "submission_ref": int(existing["ref"]),
        "public_lb": float(existing["public_lb"]),
        "kernel": str(existing["kernel"]),
        "kernel_version": int(existing["kernel_version"]),
        "source": source,
    }


def candidate_manifest_frame(config: Mapping[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(candidate_entries(config))
    frame["candidate_order"] = np.arange(len(frame), dtype=np.int16)
    batch_by_id = {
        **{candidate_id: "batch_1" for candidate_id in PAIR_IDS},
        **{
            candidate_id: "batch_2"
            for candidate_id in NORMAL_SUBMISSION_IDS[5:]
        },
        "exp226_k16": "existing_equivalence_gate",
        "likpf_mean": "existing_equivalence_gate",
        FIXED_ID: "existing_exact",
    }
    frame["batch"] = frame["id"].map(batch_by_id)
    return frame[
        [
            "candidate_order",
            "id",
            "kind",
            "formula",
            "oof_rmse",
            "lb_action",
            "batch",
            "public_lb",
            "submission_ref",
        ]
    ]


# %% [markdown]
# ## 6. Setup and frozen configuration preflight

# %%
IMPORT_ONLY = os.environ.get(IMPORT_ONLY_ENV) == "1"

if not IMPORT_ONLY:
    started = time.time()
    config = load_config()
    scientific_contract = validate_candidate_contract(
        config,
        require_run_approval=True,
    )
    paths = build_runtime_paths()
    sample = pd.read_csv(paths.sample_submission, dtype={"id": str})
    sample_audit = validate_sample_submission(sample)
    selected_candidate = str(nested(config, "execution.selected_candidate"))
    selected_entry = {
        item["id"]: item for item in candidate_entries(config)
    }[selected_candidate]
    manifest_frame = candidate_manifest_frame(config)
    manifest_path = paths.artifacts_dir / "frozen_candidate_manifest.csv"
    manifest_frame.to_csv(manifest_path, index=False)

    display(
        {
            "experiment": EXPERIMENT_NAME,
            "route": nested(config, "experiment.route"),
            "status": nested(config, "experiment.status"),
            "selected_candidate": selected_candidate,
            "selected_kind": selected_entry["kind"],
            "selected_formula": selected_entry["formula"],
            "selected_oof_rmse": selected_entry["oof_rmse"],
            "candidate_version_label": nested(
                config, "execution.candidate_version_label"
            ),
            "candidate_count": len(manifest_frame),
            "batch_1_count": len(PAIR_IDS),
            "batch_2_count": len(NORMAL_SUBMISSION_IDS[5:]),
            "sample": sample_audit,
            "device": nested(config, "runtime.device"),
            "gpu": nested(config, "runtime.kaggle.enable_gpu"),
            "internet": nested(config, "runtime.kaggle.enable_internet"),
            "training_variants": 0,
            "model_configs": 0,
            "trained_folds": 0,
            "boosters": 0,
        }
    )
    display(manifest_frame)


# %% [markdown]
# ## 7. Trusted source and provenance resolution

# %%
if not IMPORT_ONLY:
    generator = nested(config, "generator_contract")
    source_specs = {
        "exp073_pf_replay": generator["pf_replay"],
        "exp209_exact_hmm": generator["exact_hmm"],
        "exp223_selfgr_hmm": generator["selfgr_hmm_a070"],
        "exp226_k16": generator["exp226_k16"],
    }
    resolved_sources: dict[str, Path] = {}
    source_audit: dict[str, dict[str, Any]] = {}
    expected_source_sha = nested(config, "data.parent_source_sha256")
    for source_id, spec in source_specs.items():
        path = resolve_unique_source(
            str(spec["source_filename"]),
            str(spec["source_path_token"]),
        )
        resolved_sources[source_id] = path
        source_audit[source_id] = source_record(
            path,
            expected_sha256=str(expected_source_sha[source_id]),
        )

    stage0_spec = generator["stage0_manifest"]
    stage0_manifest_path = resolve_unique_source(
        str(stage0_spec["filename"]),
        str(stage0_spec["path_token"]),
    )
    source_audit["exp263_stage0_manifest"] = source_record(
        stage0_manifest_path,
        expected_sha256=str(nested(config, "data.parent_stage0_manifest_sha256")),
    )

    k16_config_path = (
        resolved_sources["exp226_k16"].parent
        / str(generator["exp226_k16"]["source_config_filename"])
    )
    source_audit["exp226_source_config"] = source_record(
        k16_config_path,
        expected_sha256=str(nested(config, "data.exp226_source_config_sha256")),
    )
    k16_config = yaml.safe_load(k16_config_path.read_text()) or {}

    trusted_source_dir = paths.artifacts_dir / "trusted_upstream_sources"
    copy_trusted_source(
        resolved_sources["exp073_pf_replay"],
        trusted_source_dir,
        "exp434_trusted_exp073_pf_replay",
    )
    copy_trusted_source(
        resolved_sources["exp209_exact_hmm"],
        trusted_source_dir,
        "exp434_trusted_exp209_exact_hmm",
    )
    copy_trusted_source(
        resolved_sources["exp223_selfgr_hmm"],
        trusted_source_dir,
        "exp434_trusted_exp223_selfgr_hmm",
    )
    copy_trusted_source(
        resolved_sources["exp226_k16"],
        trusted_source_dir,
        "exp434_trusted_exp226_k16",
    )
    sys.path.insert(0, str(trusted_source_dir))
    import exp434_trusted_exp073_pf_replay as pf_module
    import exp434_trusted_exp209_exact_hmm as exact_module
    import exp434_trusted_exp223_selfgr_hmm as selfgr_module
    import exp434_trusted_exp226_k16 as k16_module

    display(source_audit)


# %% [markdown]
# ## 8. Hidden-safe six-primitive regeneration

# %%
if not IMPORT_ONLY:
    pf_config = generator["pf_replay"]
    pf_module.configure_public_runtime(
        data_dir=paths.data_root,
        output_dir=paths.artifacts_dir / "pf_replay",
        n_jobs=int(pf_config["n_jobs"]),
        pf_seeds=int(pf_config["pf_seeds"]),
        pf_particles=int(pf_config["pf_particles"]),
        fast=bool(pf_config["fast"]),
        use_gpu=str(pf_config["use_gpu"]),
    )
    pf_frame, pf_meta = pf_module.build_replay_test_frame()
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
    missing_pf = required_pf - set(pf_frame)
    if missing_pf:
        raise ValueError(
            f"exp073 raw-test replay columns missing: {sorted(missing_pf)}"
        )

    k16_frame, k16_summary = generate_k16_primitive(
        k16_module,
        train_dir=paths.train_dir,
        test_dir=paths.test_dir,
        config=k16_config,
    )
    exact_frame = generate_hmm_primitive(
        list_well_ids=exact_module.list_well_ids,
        load_well=exact_module.load_well,
        run_hmm2=exact_module.run_hmm2,
        test_dir=paths.test_dir,
        hmm_params=dict(generator["exact_hmm"]["params"]),
    )
    selfgr_frame = generate_hmm_primitive(
        list_well_ids=selfgr_module.list_well_ids,
        load_well=selfgr_module.load_well,
        run_hmm2=selfgr_module.run_hmm2,
        test_dir=paths.test_dir,
        hmm_params=dict(generator["exact_hmm"]["params"]),
        self_gr=dict(generator["selfgr_hmm_a070"]),
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
    _, primitive_audits = primitive_audit(primitive_frames)
    display(pd.DataFrame.from_dict(primitive_audits, orient="index"))


# %% [markdown]
# ## 9. Twelve-candidate formula bank and equivalence gates

# %%
if not IMPORT_ONLY:
    candidate_bank, formula_parity = build_candidate_bank(primitive_frames)
    if len(candidate_bank) != len(sample):
        raise ValueError("candidate bank row count differs from sample submission")
    if set(candidate_bank["id"].astype(str)) != set(sample["id"].astype(str)):
        raise ValueError("candidate bank ID set differs from sample submission")

    bank_path = paths.artifacts_dir / "frozen_candidate_bank.parquet"
    candidate_bank.to_parquet(bank_path, index=False, compression="zstd")

    reference_config = nested(config, "data.exposed_reference")
    exposed_reference_path = resolve_unique_source(
        str(reference_config["filename"]),
        str(reference_config["path_token"]),
    )
    exposed_reference_gate = compare_exposed_reference(
        candidate_bank,
        primitive_frames,
        exposed_reference_path,
        config,
    )

    parent_formula_path = resolve_unique_source(
        "current_test_formula_parity.parquet",
        "exp263-last-anchor-pair-cache-inference",
    )
    parent_formula_gate = compare_parent_formula_bank(
        candidate_bank,
        parent_formula_path,
        config,
    )

    existing_artifacts = nested(config, "existing_lb.artifact_sources")
    existing_equivalence_gates = {}
    for existing_candidate in (
        "exp226_k16",
        "likpf_mean",
        FIXED_ID,
    ):
        artifact_spec = existing_artifacts[existing_candidate]
        artifact_path = resolve_unique_source(
            str(artifact_spec["filename"]),
            str(artifact_spec["path_token"]),
        )
        existing_equivalence_gates[existing_candidate] = existing_submission_gate(
            candidate_bank,
            existing_candidate,
            artifact_path,
            config,
        )

    formula_audit = pd.DataFrame(
        [
            {
                "candidate_id": candidate_id,
                "kind": EXPECTED_KINDS[candidate_id],
                "formula": EXPECTED_FORMULAS[candidate_id],
                "max_abs_parent_arithmetic_parity_ft": formula_parity.get(
                    candidate_id, 0.0
                ),
                "rows": len(candidate_bank),
                "wells": int(candidate_bank["well"].nunique()),
                "finite_rows": int(
                    np.isfinite(candidate_bank[candidate_id]).sum()
                ),
                "fallback_rows": 0,
                "prediction_content_sha256": frame_content_sha256(
                    candidate_bank[["id", candidate_id]].rename(
                        columns={candidate_id: "prediction"}
                    )
                ),
            }
            for candidate_id in CANDIDATE_IDS
        ]
    )
    formula_audit_path = paths.artifacts_dir / "candidate_formula_audit.csv"
    formula_audit.to_csv(formula_audit_path, index=False)
    display(formula_audit)
    display(
        {
            "exposed_reference_gate": exposed_reference_gate,
            "parent_formula_gate": parent_formula_gate,
            "existing_equivalence_gates": existing_equivalence_gates,
        }
    )


# %% [markdown]
# ## 10. Selected candidate and submission generation

# %%
if not IMPORT_ONLY:
    submission = build_submission(sample, candidate_bank, selected_candidate)
    submission.to_csv(paths.submission, index=False)
    if len(submission) != len(sample) or not submission["id"].equals(
        sample["id"].astype(str).reset_index(drop=True)
    ):
        raise ValueError("written submission row/order contract failed")

    selected_prediction_sha = frame_content_sha256(
        submission.rename(columns={"tvt": "prediction"})
    )
    submission_sha = sha256_file(paths.submission)
    selected_stats = {
        "rows": len(submission),
        "wells": int(candidate_bank["well"].nunique()),
        "fallback_rows": 0,
        "min": float(submission["tvt"].min()),
        "max": float(submission["tvt"].max()),
        "mean": float(submission["tvt"].mean()),
        "std": float(submission["tvt"].std()),
    }
    display(submission.head(20))
    display(selected_stats)


# %% [markdown]
# ## 11. Metrics, SHA, manifests, and generated artifacts

# %%
if not IMPORT_ONLY:
    candidate_version_manifest = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": "generated_pending_submit_check_and_submission_approval",
        "candidate_version_label": nested(
            config, "execution.candidate_version_label"
        ),
        "selected_candidate": selected_candidate,
        "selected_kind": selected_entry["kind"],
        "selected_formula": selected_entry["formula"],
        "selected_oof_rmse": float(selected_entry["oof_rmse"]),
        "kernel": nested(config, "runtime.kaggle.kernel_id"),
        "kernel_version": nested(config, "execution.kernel_version"),
        "canonical_notebook_sha256": nested(
            config, "execution.canonical_notebook_sha256"
        ),
        "prepared_package_notebook_sha256": (
            "recorded_externally_after_prepare_to_avoid_self_referential_hash"
        ),
        "scientific_contract_sha256": scientific_contract["sha256"],
        "config_sha256": sha256_file(PACKAGE_DIR / "config.yaml"),
        "candidate_manifest_sha256": sha256_file(manifest_path),
        "candidate_bank_sha256": sha256_file(bank_path),
        "formula_audit_sha256": sha256_file(formula_audit_path),
        "source_audit": source_audit,
        "primitive_audit": primitive_audits,
        "formula_parity": formula_parity,
        "exposed_reference_gate": exposed_reference_gate,
        "parent_formula_gate": parent_formula_gate,
        "existing_equivalence_gates": existing_equivalence_gates,
        "selected_prediction_content_sha256": selected_prediction_sha,
        "submission_sha256": submission_sha,
        "prediction_stats": selected_stats,
        "submit_check": "pending",
        "competition_submission": "not_approved_not_started",
        "public_lb": None,
        "deterministic_anchor": False,
    }
    version_manifest_path = (
        paths.artifacts_dir / "candidate_version_manifest.json"
    )
    version_manifest_text = (
        json.dumps(
            candidate_version_manifest,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n"
    )
    version_manifest_path.write_text(version_manifest_text)

    metrics = {
        "experiment": EXPERIMENT_NAME,
        "route": "pf_beam",
        "status": "inference_candidate_generated_pending_submit_check",
        "selected_candidate": selected_candidate,
        "candidate_version_label": nested(
            config, "execution.candidate_version_label"
        ),
        "selected_oof_rmse": float(selected_entry["oof_rmse"]),
        "rows": len(submission),
        "wells": int(candidate_bank["well"].nunique()),
        "primitive_count": 6,
        "pair_count": 5,
        "fixed_count": 1,
        "candidate_count": 12,
        "model_configs": 0,
        "trained_folds": 0,
        "boosters": 0,
        "parent_retraining": 0,
        "fallback_rows": 0,
        "sample_audit": sample_audit,
        "source_audit": source_audit,
        "primitive_audit": primitive_audits,
        "formula_parity": formula_parity,
        "exposed_reference_gate": exposed_reference_gate,
        "parent_formula_gate": parent_formula_gate,
        "existing_equivalence_gates": existing_equivalence_gates,
        "pf_generation": pf_meta,
        "exp226_generation": k16_summary,
        "scientific_contract_sha256": scientific_contract["sha256"],
        "candidate_manifest_sha256": sha256_file(manifest_path),
        "candidate_bank_sha256": sha256_file(bank_path),
        "formula_audit_sha256": sha256_file(formula_audit_path),
        "candidate_version_manifest_sha256": sha256_file(version_manifest_path),
        "prediction_content_sha256": selected_prediction_sha,
        "submission_sha256": submission_sha,
        "prediction_stats": selected_stats,
        "runtime_seconds": round(time.time() - started, 3),
        "model_sha": "not_applicable_no_training",
        "submit_check": "pending",
        "competition_submission": "not_approved_not_started",
        "public_lb": None,
        "deterministic_anchor": False,
    }
    metrics_text = (
        json.dumps(metrics, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    paths.metrics.write_text(metrics_text)
    (paths.artifacts_dir / "inference_metrics.json").write_text(metrics_text)

    print("Generated artifacts:")
    for generated_path in (
        manifest_path,
        bank_path,
        formula_audit_path,
        version_manifest_path,
        paths.metrics,
        paths.submission,
    ):
        print(f"- {generated_path} ({generated_path.stat().st_size} bytes)")
